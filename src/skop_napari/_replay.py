"""The Replay panel: a recipe, run again on the layers the user chooses.

A recipe holds op IDs, parameters and how its steps connect, never data. To
replay one is to say which layer stands for each of its inputs, which of its
steps to produce, and to run it. The results arrive as layers, and every run
is recorded in the viewer's session like any other, so the Record panel shows
the replayed steps as they land.

What a recipe needs, and whether it can run, is ``skop_core.prepare``'s
answer; turning chosen layers into what a replay reads is the host's
(``_host.Host.bind``). This module collects the choices and keeps the viewer
responsive: the replay runs on a worker, and each step is drawn as it lands.

See docs/spec/replay-onto-layers.md for the plan this builds.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import napari.viewer
from napari.qt.threading import create_worker
from napari.utils.notifications import show_error
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import ensure_main_thread

from skop import OpSpec
from skop_core import Prepared, Replayed, RootNeed, load, prepare

from ._record import _short_op
from ._roles import input_layer_types
from ._session import Session, session_for

_log = logging.getLogger("skop_napari")

#: What an array input's dropdown says while no layer is chosen.
_UNCHOSEN = "(choose a layer)"


class ReplayPanel(QWidget):
    """Runs a recipe on chosen layers, and draws and records what it makes."""

    #: Emitted on the GUI thread when a replay has finished, however it ended.
    finished = Signal()

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer | None = None,
        specs: list[OpSpec] | None = None,
        registry: dict[str, Any] | None = None,
        root: str | Path | None = None,
    ) -> None:
        # NB: named napari_viewer for the reason the Ops panel gives.
        if napari_viewer is None:
            raise ValueError(
                "The Replay panel draws into a viewer and needs one. napari "
                "supplies it to a parameter named 'napari_viewer'."
            )
        super().__init__()
        self._viewer = napari_viewer
        self._session: Session = session_for(
            napari_viewer, specs=specs, registry=registry, root=root
        )
        self._host = self._session.host
        self._recipe: dict[str, Any] | None = None
        self._source = ""
        self._prepared: Prepared | None = None
        self._targets: list[str] = []
        #: What the user chose for each input: a layer, or a path's text.
        self._bindings: dict[str, Any] = {}
        #: What each array input's dropdown offers, in its order.
        self._offered: dict[str, list[Any]] = {}
        self._running: Any = None
        self._task: Any = None
        self._cancel_asked = False
        #: What the last replay came to, said once it is over.
        self._outcome = ""

        self._open = QPushButton("Open recipe...")
        self._title = QLabel("No recipe open.")
        self._title.setWordWrap(True)
        self._problems = QLabel("")
        self._problems.setWordWrap(True)
        self._problems.setStyleSheet("color: #d93025")
        self._caution = QLabel("")
        self._caution.setWordWrap(True)
        self._inputs = QFormLayout()
        self._steps = QListWidget()
        self._steps.setToolTip(
            "Tick the steps to produce. Everything they need runs too."
        )
        self._every = QCheckBox("Show every step, not only the ticked ones")
        self._run = QPushButton("Run")
        self._cancel = QPushButton("Cancel")
        self._cancel.setVisible(False)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        self._status.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(self._open)
        top.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addWidget(self._run)
        buttons.addWidget(self._cancel)
        buttons.addStretch(1)
        column = QVBoxLayout(self)
        column.addLayout(top)
        column.addWidget(self._title)
        column.addWidget(self._problems)
        column.addWidget(self._caution)
        column.addWidget(QLabel("<b>Inputs</b>"))
        column.addLayout(self._inputs)
        column.addWidget(QLabel("<b>Produce</b>"))
        column.addWidget(self._steps, 1)
        column.addWidget(self._every)
        column.addLayout(buttons)
        column.addWidget(self._progress)
        column.addWidget(self._status)

        self._open.clicked.connect(self._open_dialog)
        self._run.clicked.connect(self.run)
        self._cancel.clicked.connect(self.cancel)
        self._steps.itemChanged.connect(self._ticked)
        layers = napari_viewer.layers.events
        for event in (layers.inserted, layers.removed, layers.renamed):
            event.connect(self._layers_changed)
        self._refresh()

    # -- the recipe ------------------------------------------------------

    def load(self, recipe: dict[str, Any] | str | Path) -> None:
        """Open a recipe: a file, or one already read."""
        if isinstance(recipe, (str, Path)):
            self._source = Path(recipe).name
            recipe = load(recipe)
        else:
            self._source = "recipe"
        self._recipe = recipe
        self._targets = list(recipe["targets"])
        self._bindings = {}
        self._rebuild_steps()
        self._prepare()

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recipe", "", "Recipe (*.json)"
        )
        if not path:
            return
        try:
            self.load(path)
        except Exception as exc:  # noqa: BLE001 -- said, not raised
            self._status.setText(f"Could not open {Path(path).name}: {exc}")

    def _prepare(self) -> None:
        """Work out what the ticked steps need, keeping choices made so far."""
        if self._recipe is None:
            self._prepared = None
        else:
            self._prepared = prepare(
                self._recipe, self._host.registry, self._targets or []
            )
        self._refresh()

    # -- the steps to produce --------------------------------------------

    def _rebuild_steps(self) -> None:
        self._steps.blockSignals(True)
        try:
            self._steps.clear()
            for step, entry in (self._recipe or {}).get("steps", {}).items():
                item = QListWidgetItem(f"{step}  ({_short_op(entry['op'])})")
                item.setData(Qt.ItemDataRole.UserRole, step)
                item.setFlags(
                    Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if step in self._targets
                    else Qt.CheckState.Unchecked
                )
                self._steps.addItem(item)
        finally:
            self._steps.blockSignals(False)

    def _ticked(self, item: QListWidgetItem) -> None:
        self.set_targets(
            [
                self._steps.item(row).data(Qt.ItemDataRole.UserRole)
                for row in range(self._steps.count())
                if self._steps.item(row).checkState() == Qt.CheckState.Checked
            ]
        )

    def set_targets(self, targets: list[str]) -> None:
        """Choose which of the recipe's steps to produce."""
        self._targets = list(targets)
        self._prepare()

    # -- the inputs ------------------------------------------------------

    def bind(self, name: str, value: Any) -> None:
        """Say what stands for an input: a layer, or a path for a file."""
        self._bindings[name] = value
        self._refresh()

    def _layers_changed(self, event: Any = None) -> None:
        # A removed layer no longer stands for anything; a rename changes
        # nothing, since a binding is the layer itself, not its name.
        alive = set(map(id, self._viewer.layers))
        self._bindings = {
            name: value
            for name, value in self._bindings.items()
            if isinstance(value, (str, Path)) or id(value) in alive
        }
        self._refresh()

    def candidates(self, need: RootNeed) -> list[Any]:
        """The layers that may stand for an input: the right type and axes."""
        kinds = input_layer_types(need.role)
        return [
            layer
            for layer in self._viewer.layers
            if type(layer).__name__.lower() in kinds
            and (need.ndim is None or layer.ndim == need.ndim)
        ]

    def _settle_bindings(self) -> None:
        """Choose for each input nobody has chosen for, when the choice is clear.

        The layer called what the input is called, else the only layer that
        fits. Anything less clear is left for the user.
        """
        for need in self._needs():
            if need.is_path or need.name in self._bindings:
                continue
            offered = self.candidates(need)
            named = [layer for layer in offered if layer.name == need.name]
            if named:
                self._bindings[need.name] = named[0]
            elif len(offered) == 1:
                self._bindings[need.name] = offered[0]

    def _rebuild_inputs(self) -> None:
        while self._inputs.rowCount():
            self._inputs.removeRow(0)
        self._offered = {}
        for need in self._needs():
            label = need.name
            if need.kept_for:
                label = f"{need.name} (kept pixels for {need.kept_for})"
            tip = f"Read by {', '.join(need.read_by)}."
            if need.is_path:
                self._inputs.addRow(label, self._path_row(need, tip))
                continue
            tip += f" Wants {need.ndim} axes."
            combo = QComboBox()
            offered = self.candidates(need)
            self._offered[need.name] = offered
            combo.addItem(_UNCHOSEN)
            for layer in offered:
                combo.addItem(layer.name)
            chosen = self._bindings.get(need.name)
            if chosen is not None and chosen in offered:
                combo.setCurrentIndex(1 + offered.index(chosen))
            combo.setToolTip(tip)
            combo.activated.connect(partial(self._chosen, need.name))
            self._inputs.addRow(label, combo)

    def _path_row(self, need: RootNeed, tip: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        field = QLineEdit(str(self._bindings.get(need.name, "")))
        field.setPlaceholderText(f"a file like {need.path_hint}")
        field.setToolTip(tip)
        field.editingFinished.connect(
            lambda: self._path_chosen(need.name, field.text())
        )
        browse = QPushButton("...")
        browse.clicked.connect(partial(self._browse, need, field))
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _browse(self, need: RootNeed, field: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"Choose {need.path_hint}")
        if path:
            field.setText(path)
            self._path_chosen(need.name, path)

    def _path_chosen(self, name: str, text: str) -> None:
        if text.strip():
            self._bindings[name] = text.strip()
        else:
            self._bindings.pop(name, None)
        self._refresh_state()

    def _chosen(self, name: str, index: int) -> None:
        if index == 0:
            self._bindings.pop(name, None)
        else:
            self._bindings[name] = self._offered[name][index - 1]
        self._refresh_state()

    def _needs(self) -> list[RootNeed]:
        return [] if self._prepared is None else list(self._prepared.roots)

    # -- what the panel says ---------------------------------------------

    def _refresh(self) -> None:
        self._settle_bindings()
        self._rebuild_inputs()
        self._refresh_state()

    def _refresh_state(self) -> None:
        recipe = self._recipe
        if recipe is None:
            self._title.setText("No recipe open.")
        else:
            self._title.setText(
                f"<b>{self._source}</b>: {len(recipe['steps'])} steps, "
                f"producing {', '.join(recipe['targets'])} by default."
            )
        prepared = self._prepared
        self._problems.setText(" ".join(prepared.problems) if prepared else "")
        self._caution.setText(_caution(recipe))
        reason = self.why_not()
        self._run.setEnabled(reason is None)
        self._run.setToolTip(reason or "Run the recipe on these layers.")
        if self._running is None:
            self._status.setText(reason or "Ready.")

    def why_not(self) -> str | None:
        """Why Run is not possible now, in words; None when it is."""
        if self._running is not None:
            return "A replay is running."
        if self._recipe is None or self._prepared is None:
            return "Open a recipe to replay."
        if not self._targets:
            return "Tick at least one step to produce."
        if self._prepared.problems:
            return " ".join(self._prepared.problems)
        unbound = [n.name for n in self._needs() if n.name not in self._bindings]
        if unbound:
            return f"Choose what stands for {', '.join(unbound)}."
        return None

    # -- running ---------------------------------------------------------

    def run(self) -> None:
        """Replay the recipe on the chosen layers, off the GUI thread."""
        reason = self.why_not()
        prepared = self._prepared
        if reason is not None or prepared is None:
            self._status.setText(reason or "Nothing to run.")
            return
        try:
            roots = self._host.bind(prepared, self._bindings)
            prepared.check(roots)
        except Exception as exc:  # noqa: BLE001 -- said, not raised
            self._status.setText(f"{type(exc).__name__}: {exc}")
            return

        drawn = set(prepared.steps if self._every.isChecked() else prepared.targets)
        self._cancel_asked = False
        self._task = None
        self._outcome = ""
        self._progress.setRange(0, len(prepared.steps))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._cancel.setVisible(True)
        self._cancel.setEnabled(True)
        work = partial(
            prepared.run,
            roots,
            self._session.recorder,
            on_step=ensure_main_thread(self._stepping),
            on_done=ensure_main_thread(partial(self._landed, drawn)),
            cancelled=lambda: self._cancel_asked,
            on_start=self._started,
        )
        # NB: connected through _connect, for the reason the Ops panel gives:
        # the worker starts at once, and an error with no handler is re-raised
        # into the event loop.
        self._running = create_worker(
            work,
            _start_thread=False,
            _connect={
                "returned": self._returned,
                "errored": self._errored,
                "finished": self._finished,
            },
        )
        self._refresh_state()
        self._status.setText(f"Replaying {len(prepared.steps)} steps...")
        self._running.start()

    def cancel(self) -> None:
        """Stop after the step running now, and ask that step to stop too."""
        self._cancel_asked = True
        self._cancel.setEnabled(False)
        task = self._task
        if task is not None:
            task.cancel()

    def _started(self, task: Any) -> None:
        self._task = task
        if self._cancel_asked:
            task.cancel()

    def _stepping(self, index: int, total: int, step: str, session: str) -> None:
        self._progress.setValue(index)
        named = step if session == step else f"{step} (as {session})"
        self._status.setText(f"Step {index + 1} of {total}: {named}")

    def _landed(self, drawn: set[str], step: str, session: str) -> None:
        """A step was recorded: draw it if it is one to show, and say so."""
        self._progress.setValue(self._progress.value() + 1)
        if step in drawn:
            try:
                self._host.show(session)
            except KeyError:
                _log.info("Nothing to draw for %r", session)
        self._session.changed.emit()

    def _returned(self, done: Replayed) -> None:
        ran = len(done.steps)
        self._outcome = (
            f"Cancelled after {ran} steps."
            if done.cancelled
            else f"Replayed {ran} steps."
        )

    def _errored(self, exc: Exception) -> None:
        # The steps before the failure were recorded and drawn as they
        # landed; the failed one is recorded as failed.
        _log.error("Replay failed", exc_info=exc)
        show_error(f"Replay failed: {exc}")
        self._outcome = f"{type(exc).__name__}: {exc}"

    def _finished(self) -> None:
        self._running = None
        self._task = None
        self._progress.setVisible(False)
        self._cancel.setVisible(False)
        self._refresh_state()
        self._status.setText(self._outcome)
        self.finished.emit()


def _caution(recipe: dict[str, Any] | None) -> str:
    """What the recipe warns about, said to someone about to run it."""
    if recipe is None:
        return ""
    caution = recipe.get("caution", {})
    said = []
    if caution.get("edited"):
        said.append(
            f"{', '.join(caution['edited'])} had been edited by hand when "
            f"this was recorded; a replay recomputes it without the edit."
        )
    if caution.get("affected"):
        said.append(
            f"{', '.join(caution['affected'])} read an edited result, so "
            f"may come out differently."
        )
    return " ".join(said)
