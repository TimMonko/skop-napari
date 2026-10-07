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

import contextlib
import logging
from functools import partial
from pathlib import Path
from typing import Any

import napari.viewer
from magicgui.widgets import Container
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
    QToolBox,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import ensure_main_thread

from skop import NDARRAY, PATH, OpSpec, ParamSpec, type_spec
from skop_core import Prepared, Replayed, RootNeed, load, prepare, revive

from ._axes import resolve
from ._choices import is_workflow_plumbing
from ._record import _short_op
from ._roles import annotation_for, input_layer_types, value_for
from ._session import Session, session_for
from ._widget import Inputs, build_inputs

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
        #: Each planned step's form, and the values it started from.
        self._forms: dict[str, tuple[Inputs, dict[str, Any], Container]] = {}
        #: What the user changed, by step, so a rebuilt form keeps it.
        self._edits: dict[str, dict[str, Any]] = {}
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
        self._params = QToolBox()
        self._params.setToolTip(
            "Each step's parameters, as the recipe recorded them. What you "
            "change here is what the replay runs with, and is recorded."
        )
        self._every = QCheckBox("Show every step, not only the ticked ones")
        self._run = QPushButton("Run")
        self._each = QPushButton("Run on each selected layer")
        self._each.setToolTip(
            "One replay per selected layer, each standing for the recipe's "
            "image input. A layer that fails does not stop the rest."
        )
        self._last = QPushButton("Open last export")
        self._last.setToolTip("The recipe the Record panel exported last.")
        self._cancel = QPushButton("Cancel")
        self._cancel.setVisible(False)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        self._status.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(self._open)
        top.addWidget(self._last)
        top.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addWidget(self._run)
        buttons.addWidget(self._each)
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
        column.addWidget(QLabel("<b>Parameters</b>"))
        column.addWidget(self._params, 2)
        column.addWidget(self._every)
        column.addLayout(buttons)
        column.addWidget(self._progress)
        column.addWidget(self._status)

        self._open.clicked.connect(self._open_dialog)
        self._run.clicked.connect(self.run)
        self._each.clicked.connect(lambda: self.run_on_each())
        self._last.clicked.connect(self._open_last)
        self._session.changed.connect(self._last_export_changed)
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
        self._edits = {}
        self._rebuild_steps()
        self._prepare()

    def _open_last(self) -> None:
        recipe = self._session.last_recipe
        if recipe is not None:
            self.load(recipe)
            self._source = "the last export"
            self._refresh_state()

    def _last_export_changed(self) -> None:
        try:
            self._last.setEnabled(self._session.last_recipe is not None)
        except RuntimeError:
            # The panel was closed; the session outlived it.
            self._session.changed.disconnect(self._last_export_changed)

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
        self._rebuild_params()
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
        """The layers that may stand for an input: the right type and axes.

        More axes than were recorded are fine for an input every reader of
        which declares its axes: the replay runs plane by plane over them.
        """
        kinds = input_layer_types(need.role)
        return [
            layer
            for layer in self._viewer.layers
            if type(layer).__name__.lower() in kinds and need.fits(layer.ndim)
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
            tip += f" Wants {need.ndim} axes"
            tip += (
                ", or more: the extra ones are run over, plane by plane."
                if need.plannable
                else "."
            )
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

    # -- each step's parameters -------------------------------------------

    def _rebuild_params(self) -> None:
        """One form per step that will run, filled in from the recipe.

        The same forms the Ops panel builds (``build_inputs``), so a recipe's
        parameters look and behave as they do when the op is run by hand.
        What a form cannot show -- a step's inputs, which are bound above,
        a file, or a workflow's choice of op -- keeps the recipe's value.
        """
        while self._params.count():
            self._params.removeItem(0)
        self._forms = {}
        prepared = self._prepared
        if prepared is None or prepared.problems or self._recipe is None:
            return
        for step in prepared.steps:
            entry = self._recipe["steps"][step]
            spec = OpSpec.from_op(self._host.registry[entry["op"]])
            inputs = build_inputs(
                spec,
                annotation_for,
                value_for,
                skip=partial(_not_a_setting, entry),
            )
            if not inputs.widgets:
                continue
            base = _recorded(spec, entry)
            edits = self._edits.get(step, {})
            for widget in inputs.widgets:
                if widget.name in edits or widget.name in base:
                    with contextlib.suppress(Exception):
                        widget.value = edits.get(widget.name, base.get(widget.name))
                widget.changed.connect(partial(self._param_changed, step, widget.name))
            container = Container(widgets=inputs.widgets)
            self._forms[step] = (inputs, base, container)
            self._params.addItem(container.native, self._form_title(step))

    def _param_changed(self, step: str, name: str, value: Any) -> None:
        self._edits.setdefault(step, {})[name] = value
        index = list(self._forms).index(step)
        self._params.setItemText(index, self._form_title(step))

    def _form_title(self, step: str) -> str:
        assert self._recipe is not None
        title = f"{step}  ({_short_op(self._recipe['steps'][step]['op'])})"
        changed = len(self.overrides().get(step, {})) if step in self._forms else 0
        return f"{title}  - {changed} changed" if changed else title

    def set_param(self, step: str, name: str, value: Any) -> None:
        """Change one step's parameter, as editing its form would."""
        inputs = self._forms[step][0]
        next(w for w in inputs.widgets if w.name == name).value = value

    def overrides(self) -> dict[str, dict[str, Any]]:
        """The parameters the user changed from the recipe's, by step."""
        found: dict[str, dict[str, Any]] = {}
        for step, (inputs, base, _) in self._forms.items():
            changed = {
                name: value
                for name, value in inputs.values().items()
                if name not in base or base[name] != value
            }
            if changed:
                found[step] = changed
        return found

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
        each = self.why_not_each()
        self._each.setEnabled(each is None)
        if each is not None:
            self._each.setToolTip(each)
        self._last.setEnabled(self._session.last_recipe is not None)
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
        if reason is not None:
            self._status.setText(reason)
            return
        self._start([("", dict(self._bindings))])

    def run_on_each(self, layers: list[Any] | None = None) -> None:
        """Replay once per layer, each standing for the recipe's image input.

        Batch, with layers for files: one replay per layer, carrying on past a
        layer that fails and saying which did (decision 19). ``layers``
        defaults to the viewer's selection; layers that do not fit the input
        are left out.
        """
        reason = self.why_not_each()
        if reason is not None:
            self._status.setText(reason)
            return
        need = self._single_array()
        assert need is not None
        chosen = list(self._viewer.layers.selection) if layers is None else layers
        offered = self.candidates(need)
        fitting = [layer for layer in chosen if layer in offered]
        if not fitting:
            self._status.setText(
                f"None of the selected layers fits {need.name!r}: select the "
                f"layers to run on."
            )
            return
        others = {k: v for k, v in self._bindings.items() if k != need.name}
        self._start([(layer.name, {**others, need.name: layer}) for layer in fitting])

    def why_not_each(self) -> str | None:
        """Why Run on each is not possible now, in words; None when it is."""
        if self._running is not None:
            return "A replay is running."
        if self._recipe is None or self._prepared is None:
            return "Open a recipe to replay."
        if not self._targets:
            return "Tick at least one step to produce."
        if self._prepared.problems:
            return " ".join(self._prepared.problems)
        arrays = [n for n in self._needs() if not n.is_path]
        if len(arrays) != 1:
            return (
                f"Running on each layer takes a recipe with one image input; "
                f"this one has {len(arrays)}."
            )
        unbound = [
            n.name for n in self._needs() if n.is_path and n.name not in self._bindings
        ]
        if unbound:
            return f"Choose what stands for {', '.join(unbound)}."
        return None

    def _single_array(self) -> RootNeed | None:
        arrays = [n for n in self._needs() if not n.is_path]
        return arrays[0] if len(arrays) == 1 else None

    def _start(self, jobs: list[tuple[str, dict[str, Any]]]) -> None:
        """Bind each job's layers here, then replay them all on a worker.

        Binding touches the layers, so it happens on the GUI thread; a job
        whose layers cannot be bound -- an edited one, say -- is reported as
        failed without stopping the others.
        """
        prepared = self._prepared
        assert prepared is not None
        bound: list[tuple[str, dict[str, Any] | None, Exception | None]] = []
        names: list[dict[str, tuple[str | None, ...]]] = []
        for label, chosen in jobs:
            names.append(
                {
                    root: resolve(value.data, value, self._viewer).axes
                    for root, value in chosen.items()
                    if hasattr(value, "data") and hasattr(value, "metadata")
                }
            )
            try:
                roots = self._host.bind(prepared, chosen)
                prepared.check(roots)
            except Exception as exc:  # noqa: BLE001 -- reported per job
                bound.append((label, None, exc))
            else:
                bound.append((label, roots, None))

        drawn = set(prepared.steps if self._every.isChecked() else prepared.targets)
        overrides = self.overrides()
        steps = len(prepared.steps)
        self._cancel_asked = False
        self._task = None
        self._outcome = ""
        self._progress.setRange(0, steps * len(jobs))
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._cancel.setVisible(True)
        self._cancel.setEnabled(True)
        recorder = self._session.recorder
        landed = ensure_main_thread(partial(self._landed, drawn))

        def work() -> list[tuple[str, Replayed | None, Exception | None]]:
            outcomes: list[tuple[str, Replayed | None, Exception | None]] = []
            for number, ((label, roots, failed), axes) in enumerate(
                zip(bound, names, strict=True)
            ):
                if self._cancel_asked:
                    break
                if roots is None:
                    outcomes.append((label, None, failed))
                    continue
                try:
                    done = prepared.run(
                        roots,
                        recorder,
                        on_step=ensure_main_thread(
                            partial(self._stepping, number, len(bound), label)
                        ),
                        on_done=landed,
                        overrides=overrides,
                        cancelled=lambda: self._cancel_asked,
                        on_start=self._started,
                        axes=axes,
                    )
                except Exception as exc:  # noqa: BLE001 -- reported per job
                    outcomes.append((label, None, exc))
                    continue
                outcomes.append((label, done, None))
                if done.cancelled:
                    break
            return outcomes

        # NB: connected through _connect, for the reason the Ops panel gives:
        # the worker starts at once, and an error with no handler is re-raised
        # into the event loop.
        self._running = create_worker(
            work,
            _start_thread=False,
            _connect={
                "returned": partial(self._returned, len(jobs)),
                "errored": self._errored,
                "finished": self._finished,
            },
        )
        self._refresh_state()
        self._status.setText(f"Replaying {steps} steps...")
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

    def _stepping(
        self,
        job: int,
        jobs: int,
        label: str,
        index: int,
        total: int,
        step: str,
        session: str,
    ) -> None:
        self._progress.setValue(job * total + index)
        named = step if session == step else f"{step} (as {session})"
        where = f"{label} ({job + 1} of {jobs}): " if jobs > 1 else ""
        self._status.setText(f"{where}step {index + 1} of {total}: {named}")

    def _landed(self, drawn: set[str], step: str, session: str) -> None:
        """A step was recorded: draw it if it is one to show, and say so."""
        self._progress.setValue(self._progress.value() + 1)
        if step in drawn:
            try:
                self._host.show(session)
            except KeyError:
                _log.info("Nothing to draw for %r", session)
        self._session.changed.emit()

    def _returned(
        self,
        jobs: int,
        outcomes: list[tuple[str, Replayed | None, Exception | None]],
    ) -> None:
        failed = [(label, exc) for label, _, exc in outcomes if exc is not None]
        for label, exc in failed:
            _log.error(
                "Replay on %s failed", label or "the chosen layers", exc_info=exc
            )
        cancelled = any(done is not None and done.cancelled for _, done, _ in outcomes)
        if jobs == 1:
            ((_, done, exc),) = outcomes or [("", None, None)]
            if exc is not None:
                # The steps before the failure were recorded and drawn as
                # they landed; the failed one is recorded as failed.
                show_error(f"Replay failed: {exc}")
                self._outcome = f"{type(exc).__name__}: {exc}"
            elif done is not None and done.cancelled:
                self._outcome = f"Cancelled after {len(done.steps)} steps."
            elif done is not None:
                self._outcome = f"Replayed {len(done.steps)} steps." + _planned(done)
            else:
                self._outcome = "Cancelled before anything ran."
            return
        ok = len(outcomes) - len(failed)
        said = [f"Ran on {ok} of {jobs} layers."]
        if cancelled or len(outcomes) < jobs:
            said.append("Cancelled before the rest.")
        if failed:
            said.append(
                "Failed: "
                + "; ".join(
                    f"{label}: {type(exc).__name__}: {exc}" for label, exc in failed
                )
            )
            show_error(f"Replay failed on {', '.join(label for label, _ in failed)}")
        self._outcome = " ".join(said)

    def _errored(self, exc: Exception) -> None:
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


def _planned(done: Replayed) -> str:
    """What a replay planned afresh, said in a sentence; empty when nothing."""
    said = [
        f"{step} ran {summary}"
        for step, plans in done.planned.items()
        for summary in plans.values()
    ]
    return f" {'; '.join(said)}." if said else ""


def _not_a_setting(entry: dict[str, Any], param: ParamSpec) -> bool:
    """Whether a parameter is bound elsewhere rather than set in a form."""
    return (
        param.name in entry["inputs"]
        or is_workflow_plumbing(param)
        or type_spec(param.type).name in (NDARRAY, PATH)
    )


def _recorded(spec: OpSpec, entry: dict[str, Any]) -> dict[str, Any]:
    """What a step ran with: its defaults, then the recipe's own values.

    As the op declares them -- an enum member, not its value -- so a form
    shows them, and a comparison with the form tells what was changed.
    """
    found = {p.name: p.default for p in spec.params if not p.required}
    try:
        found.update(revive(spec, entry["params"]))
    except KeyError:  # an op ID it cannot resolve: not a form's business
        found.update(entry["params"])
    return found


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
