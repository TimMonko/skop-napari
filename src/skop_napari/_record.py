"""The Record panel: what has been run, where it came from, and what moved.

It runs nothing. The Ops and Workflows panels run ops, as they always did, and
while this panel is recording they run them through the session's listening
runner (``_session``), so a call made anywhere, in any environment, is written
down on its way past. This panel is where the writing shows.

Top to bottom it answers four questions. *What went in?* -- the inputs, each a
dropdown of the layers that fit, so choosing another layer is how an input is
changed. *How does it fit together?* -- the graph, where pointing at a node
lights the path through it. *What is each step?* -- the table: what it read,
what it made, with which parameters, how many times, and in what state. And
*what do I do about it?* -- the detail of the selected step, and Produce.

What a host is, and what it promises, is in ``_host``: that module imports no
napari and is tested without a viewer. What a session is -- the one runner and
store a viewer's panels share -- is in ``_session``. Where the nodes go and
which are on a path is ``skop_visual``'s. This module is the part that needs
Qt: something to draw in and somewhere to press buttons.
"""

from __future__ import annotations

import html
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import napari.viewer
from napari.qt.threading import create_worker
from napari.utils.notifications import show_error
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from skop import OpSpec, Role
from skop_visual import EDITED, SIGNALS, Node, layout, lineage

from ._graph import GraphCanvas
from ._host import SKOP
from ._session import Session, session_for

_log = logging.getLogger("skop_napari")

#: The table's columns. The first is the Produce tick.
# State comes right after the step: in a narrow dock the rest scrolls, and what
# is wrong with a step is what a reader came to see.
_COLUMNS = ["", "step", "state", "in", "out", "params", "runs", "op"]
_TICK, _STEP, _STATE, _IN, _OUT, _PARAMS, _RUNS, _OP = range(len(_COLUMNS))

#: What a row is tinted when it is on the path through the selected step: the
#: steps it was made from, and the steps made from it. The same two colours the
#: graph lights its edges in.
_UPSTREAM = QColor(26, 115, 232, 60)
_DOWNSTREAM = QColor(132, 48, 206, 55)

#: The layers an input may be, by what the ops that read it take it to be.
#: Anything not listed is not an array an op can read as an input at all.
_ARRAY_LAYERS = ("image", "labels")
_LAYERS_FOR_ROLE = {Role.image: ("image",), Role.labels: ("labels",)}


def _short_op(op: str) -> str:
    """An op ID, shortened to what tells two of them apart.

    An ID is ``module:function`` and the module is long. What is worth showing
    is the namespace the op lives in -- 'mask: detect' next to 'image: smooth'
    -- so the module's own name is dropped when it repeats the function.
    """
    module, _, function = op.rpartition(":")
    tail = module.rsplit(".", 1)[-1]
    return function if tail == function else f"{tail}:{function}"


def _shown_run(node: Node) -> Any:
    """The run a row describes: the one the step stands on, else the newest."""
    return node.current or (node.runs[0] if node.runs else None)


def _reads(node: Node) -> str:
    """What a step read, as ``parameter ← where it came from``."""
    run = _shown_run(node)
    if run is None:
        return ""
    return ", ".join(
        f"{edge.param} ← {edge.source or edge.root or '?'}" for edge in run.inputs
    )


def _makes(node: Node) -> str:
    run = _shown_run(node)
    return ", ".join(run.outputs) if run is not None else ""


def _setting(value: Any) -> str:
    """A parameter's value, short enough for a cell. An op ID is an op."""
    if isinstance(value, str) and ":" in value:
        return _short_op(value)
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= 24 else text[:23] + "…"


def _settings(node: Node) -> str:
    run = _shown_run(node)
    if run is None or not run.params:
        return ""
    return ", ".join(f"{k}={_setting(v)}" for k, v in run.params.items())


def _runs(node: Node) -> str:
    failed = sum(1 for run in node.runs if run.status != "ok")
    count = len(node.runs)
    return f"{count} ({failed} failed)" if failed else str(count)


def _states(node: Node) -> str:
    flags = node.flags or ("fresh",)
    return " · ".join(SIGNALS.get(flag, flag) for flag in flags)


def _why(node: Node) -> str:
    """The words behind a state, for a tooltip and the detail."""
    reasons = {
        "stale": "An input moved after this step read it.",
        "edited": "The result was changed by hand.",
        "affected": "An edited step feeds this one.",
        "failed": "The newest run failed.",
        "missing": (
            "The layer behind "
            + ", ".join(repr(name) for name in node.missing)
            + " was removed. Nothing newer exists to catch up with, so this "
            "is not stale: it can be read, but not reproduced here."
        ),
    }
    return " ".join(reasons[flag] for flag in node.flags if flag in reasons) or (
        "Up to date."
    )


class RecordPanel(QWidget):
    """Shows what the session recorded, and brings steps up to date."""

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer | None = None,
        specs: list[OpSpec] | None = None,
        registry: dict[str, Any] | None = None,
        root: str | Path | None = None,
    ) -> None:
        # NB: named napari_viewer for the reason the Ops panel gives -- it is
        # the only check napari makes that survives a postponed annotation.
        if napari_viewer is None:
            raise ValueError(
                "The Record panel shows what a viewer's session recorded, and "
                "needs the viewer. napari supplies one to a parameter named "
                "'napari_viewer'."
            )
        super().__init__()
        self._viewer = napari_viewer
        self._session: Session = session_for(
            napari_viewer, specs=specs, registry=registry, root=root
        )
        self._host = self._session.host
        self._recorder = self._session.recorder
        #: Which steps are ticked as targets. Kept here rather than read back
        #: out of the table, so rebuilding it does not lose it.
        self._targets: set[str] = set()
        #: The step whose path is lit, or None.
        self._selected: str | None = None
        #: The worker bringing steps up to date, while there is one.
        self._producing: Any = None
        #: What the input dropdowns were last built from, so a refresh that
        #: changes nothing about them does not close one a user has open.
        self._inputs_made: tuple = ()
        self._input_layers: dict[str, list[Any]] = {}
        self._syncing = False

        self._recording = QCheckBox("Record what is run")
        self._recording.setChecked(True)
        self._recording.setToolTip(
            "While this is on, ops run from the Ops and Workflows panels are "
            "written down. They run exactly as they would otherwise."
        )

        self._inputs_form = QFormLayout()
        self._inputs_empty = QLabel("Nothing yet. An op needs a layer to read.")
        self._refresh_inputs = QPushButton("Look for new layers")
        self._refresh_inputs.setToolTip(
            "Take layers opened while nothing was recording as inputs, and "
            "notice any input whose pixels changed."
        )

        self._canvas = GraphCanvas()
        self._canvas.setToolTip(
            "Click a step to light the path through it: blue is what it was "
            "made from, purple is what is made from it."
        )

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setMinimumHeight(120)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self._table.setToolTip(
            f"The log of what was run is written to {self._session.store.log_path}."
        )

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.RichText)
        self._detail.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self._revert_button = QPushButton("Revert")
        self._keep_button = QPushButton("Keep the edit")
        self._revert_button.setVisible(False)
        self._keep_button.setVisible(False)

        self._produce = QPushButton("Produce")
        self._produce.setToolTip("Bring the ticked steps up to date.")
        self._status = QLabel("")
        self._status.setWordWrap(True)

        edit_row = QHBoxLayout()
        edit_row.addWidget(self._revert_button)
        edit_row.addWidget(self._keep_button)
        edit_row.addStretch(1)
        run_row = QHBoxLayout()
        run_row.addWidget(self._produce)
        run_row.addStretch(1)

        column = QVBoxLayout(self)
        column.addWidget(self._recording)
        column.addWidget(QLabel("<b>Inputs</b>"))
        column.addWidget(self._inputs_empty)
        column.addLayout(self._inputs_form)
        column.addWidget(self._refresh_inputs)
        column.addWidget(QLabel("<b>Graph</b>"))
        column.addWidget(self._canvas, 2)
        column.addWidget(QLabel("<b>Steps</b>"))
        column.addWidget(self._table, 3)
        column.addWidget(self._detail)
        column.addLayout(edit_row)
        column.addLayout(run_row)
        column.addWidget(self._status)

        self._recording.toggled.connect(self._record_toggled)
        self._refresh_inputs.clicked.connect(self._declare_roots)
        self._produce.clicked.connect(self._produce_)
        self._revert_button.clicked.connect(lambda: self._revert(self._selected))
        self._keep_button.clicked.connect(lambda: self._keep(self._selected))
        self._table.itemChanged.connect(self._ticked)
        self._table.itemSelectionChanged.connect(self._row_selected)
        self._canvas.picked.connect(self._node_picked)
        # Whatever ran an op -- the Ops panel, a script -- is told about here
        # only by the session saying its picture changed.
        self._session.changed.connect(self._refresh)
        self._viewer.layers.selection.events.active.connect(self._layer_selected)

        self._session.recording = self._recording.isChecked()
        self._refresh()

    # -- recording -------------------------------------------------------

    def _record_toggled(self, on: bool) -> None:
        self._session.recording = on
        self._status.setText(
            "Recording." if on else "Not recording: ops run, and are forgotten."
        )

    def _declare_roots(self) -> None:
        """Take every layer the session has not seen as an input, and re-read
        the ones it has. Idempotent, and safe to press twice."""
        self._session.declare_roots()

    # -- refreshing ------------------------------------------------------

    def _refresh(self) -> None:
        """Redraw everything from the graph, which is the only account of it.

        Read back from the log rather than remembered: a panel that kept its
        own idea of what had been run is a second source of truth, and the one
        that goes wrong.
        """
        try:
            self._rebuild()
        except RuntimeError:
            # The panel was closed but the session outlived it, so this is
            # still arriving at widgets whose Qt objects are gone.
            self._session.changed.disconnect(self._refresh)

    def _rebuild(self) -> None:
        graph = self._recorder.graph()
        view = self._host.view()
        steps = {node.step for node in view.nodes}
        self._targets &= steps
        if self._selected not in steps:
            self._selected = None

        self._rebuild_inputs(graph)
        self._rebuild_table(view, graph)
        self._draw_graph(view, graph)
        self._show_detail(view, graph)
        self._produce.setEnabled(bool(self._targets) and self._producing is None)

    # -- inputs ----------------------------------------------------------

    def _rebuild_inputs(self, graph: Any) -> None:
        """One dropdown per input, of the layers that fit it.

        An input is a name the graph knows, and the layer behind it can be
        changed: choosing another layer here stores that layer's pixels under
        the same name, so every step that read the input is stale, and Produce
        brings them up to date on the new one.
        """
        names: list[str] = []
        for root in graph.roots:
            if root.name not in names:
                names.append(root.name)

        layers = self._host.display.layers()
        offered: dict[str, list[Any]] = {}
        for name in names:
            kinds = _LAYERS_FOR_ROLE.get(self._host.root_role(name), _ARRAY_LAYERS)
            offered[name] = [
                layer for layer in layers if type(layer).__name__.lower() in kinds
            ]
        current = {name: self._host.layer_of(name) for name in names}
        made = tuple(
            (
                name,
                tuple((id(layer), layer.name) for layer in offered[name]),
                id(current[name]),
            )
            for name in names
        )
        self._inputs_empty.setVisible(not names)
        if made == self._inputs_made:
            return
        self._inputs_made = made
        self._input_layers = {}

        while self._inputs_form.rowCount():
            self._inputs_form.removeRow(0)
        for name in names:
            combo = QComboBox()
            options = list(offered[name])
            if current[name] is None:
                combo.addItem("(layer removed)")
                options.insert(0, None)
            for layer in offered[name]:
                combo.addItem(layer.name)
            if current[name] is not None and current[name] in offered[name]:
                combo.setCurrentIndex(offered[name].index(current[name]))
            self._input_layers[name] = options
            combo.setToolTip(
                f"Which layer is {name!r}. Choosing another makes everything "
                f"that read it stale."
            )
            # activated, not currentIndexChanged: only a person choosing
            # reaches it, so building the box does not rebind anything.
            combo.activated.connect(partial(self._input_chosen, name))
            self._inputs_form.addRow(name, combo)

    def _input_chosen(self, name: str, index: int) -> None:
        layer = self._input_layers[name][index]
        if layer is None:
            return
        self._session.bind_root(name, layer)
        self._status.setText(
            f"{name!r} now reads {layer.name!r}. Steps that read it are stale."
        )

    # -- the table -------------------------------------------------------

    def _rebuild_table(self, view: Any, graph: Any) -> None:
        self._syncing = True
        try:
            self._table.setRowCount(len(view.nodes))
            for row, node in enumerate(view.nodes):
                self._fill_row(row, node)
            self._tint(graph)
            self._select_row()
        finally:
            self._syncing = False

    def _fill_row(self, row: int, node: Node) -> None:
        tick = QTableWidgetItem()
        tick.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        tick.setCheckState(
            Qt.CheckState.Checked
            if node.step in self._targets
            else Qt.CheckState.Unchecked
        )
        tick.setToolTip("Produce this")
        self._table.setItem(row, _TICK, tick)

        cells = {
            _STEP: (node.step, node.step),
            _OP: (_short_op(node.op), node.op),
            _IN: (_reads(node), _reads(node)),
            _OUT: (_makes(node), _makes(node)),
            _PARAMS: (_settings(node), _settings(node)),
            _RUNS: (_runs(node), "Runs of this step; the newest good one stands."),
            _STATE: (_states(node), _why(node)),
        }
        for column, (text, tip) in cells.items():
            item = QTableWidgetItem(text)
            item.setToolTip(tip)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table.setItem(row, column, item)

    def _tint(self, graph: Any) -> None:
        """Tint the rows on the path through the selected step."""
        path = lineage(graph, self._selected) if self._selected else None
        for row in range(self._table.rowCount()):
            step = self._table.item(row, _STEP).text()
            colour = None
            if path is not None and step in path.upstream:
                colour = _UPSTREAM
            elif path is not None and step in path.downstream:
                colour = _DOWNSTREAM
            brush = QBrush(colour) if colour is not None else QBrush()
            for column in range(1, len(_COLUMNS)):
                self._table.item(row, column).setBackground(brush)

    def _select_row(self) -> None:
        self._table.clearSelection()
        for row in range(self._table.rowCount()):
            if self._table.item(row, _STEP).text() == self._selected:
                self._table.selectRow(row)

    def _ticked(self, item: QTableWidgetItem) -> None:
        if self._syncing or item.column() != _TICK:
            return
        step = self._table.item(item.row(), _STEP).text()
        self._tickle(step, item.checkState() == Qt.CheckState.Checked)

    def _tickle(self, step: str, ticked: bool) -> None:
        if ticked:
            self._targets.add(step)
        else:
            self._targets.discard(step)
        self._produce.setEnabled(bool(self._targets) and self._producing is None)
        self._status.setText(self._targets_line())

    def _targets_line(self) -> str:
        if not self._targets:
            return "Tick the steps to bring up to date, then Produce."
        return f"Targets: {', '.join(sorted(self._targets))}"

    # -- the graph, and what is selected ---------------------------------

    def _draw_graph(self, view: Any, graph: Any) -> None:
        path = lineage(graph, self._selected) if self._selected else None
        self._canvas.show_graph(layout(view), path, self._selected)

    def select(self, step: str | None) -> None:
        """Point at a step: light its path, and find its layer.

        What is lit is derived from the log (``skop_visual.lineage``), so this
        stores nothing but which step is pointed at.
        """
        if step == self._selected:
            return
        self._selected = step
        self._rebuild()
        if step is not None:
            self._select_layer(step)

    def _row_selected(self) -> None:
        if self._syncing:
            return
        rows = self._table.selectionModel().selectedRows()
        self.select(self._table.item(rows[0].row(), _STEP).text() if rows else None)

    def _node_picked(self, key: str) -> None:
        # A root has no step to point at: clicking one clears the selection.
        self.select(key if key and not key.startswith("root:") else None)

    def _select_layer(self, step: str) -> None:
        """Select the layer drawing a step, so it is findable in the viewer."""
        layer = self._session.display.layer_for(step)
        if layer is None:
            return
        self._syncing = True
        try:
            self._viewer.layers.selection.select_only(layer)
        finally:
            self._syncing = False

    def _layer_selected(self, event: Any) -> None:
        """A layer was picked in the viewer: point at the step that drew it."""
        layer = event.value
        link = getattr(layer, "metadata", {}).get(SKOP) if layer is not None else None
        if self._syncing or not link or not link.get("step"):
            return
        self._selected = link["step"]
        self._rebuild()

    # -- the detail of one step ------------------------------------------

    def _show_detail(self, view: Any, graph: Any) -> None:
        node = next((n for n in view.nodes if n.step == self._selected), None)
        edited = node is not None and EDITED in node.flags
        self._revert_button.setVisible(edited)
        self._keep_button.setVisible(edited)
        if node is None:
            self._detail.setText(
                "<i>Click a step, in the graph or the table, to see where it "
                "came from.</i>"
            )
            return

        path = lineage(graph, node.step)
        esc = html.escape
        lines = [
            f"<b>{esc(node.step)}</b> &mdash; {esc(_short_op(node.op))}",
            f"{esc(_states(node))} &mdash; {esc(_why(node))}",
        ]
        if path.upstream or path.roots:
            chain = [f"<i>{esc(r)}</i>" for r in path.roots] + [
                esc(s) for s in path.upstream
            ]
            chain.append(f"<b>{esc(node.step)}</b>")
            lines.append("from: " + " &rarr; ".join(chain))
        else:
            lines.append("from: nothing, it reads no input")
        if path.downstream:
            lines.append("feeds: " + ", ".join(esc(s) for s in path.downstream))
        run = node.current
        if run is not None and run.params:
            lines.append(
                "parameters: "
                + ", ".join(
                    f"{esc(k)} = {esc(_setting(v))}" for k, v in run.params.items()
                )
            )
        if node.runs and node.runs[0].status != "ok":
            lines.append(f"last run failed: {esc(node.runs[0].error or 'unknown')}")
        self._detail.setText("<br>".join(lines))

    # -- bringing steps up to date ---------------------------------------

    def _produce_(self) -> None:
        """Bring the ticked steps up to date, and draw what moved.

        Produce runs the backward closure of the targets and nothing else
        (decision 4), so a step whose inputs have not moved is left alone and
        costs nothing. It runs the ops through the session's runner, which may
        mean building an environment, so it happens on a worker: the viewer
        stays usable, and the layers are drawn when it is done.
        """
        if not self._targets or self._producing is not None:
            self._status.setText(self._targets_line())
            return
        targets = sorted(self._targets)
        self._produce.setEnabled(False)
        self._status.setText(f"Producing {', '.join(targets)}...")
        # NB: connected here rather than afterwards, for the reason the Ops
        # panel gives: the thread starts at once, and superqt re-raises into
        # the event loop unless an 'errored' handler was supplied this way.
        self._producing = create_worker(
            partial(self._session.produce, targets),
            _start_thread=False,
            _connect={
                "returned": self._produced,
                "errored": partial(self._produce_failed, targets),
                "finished": self._produce_finished,
            },
        )
        self._producing.start()

    def _produced(self, ran: list[str]) -> None:
        for step in ran:
            try:
                self._host.show(step)
            except Exception:  # noqa: BLE001 -- a scalar output has no layer.
                _log.info("Nothing to draw for %r", step)
        self._status.setText(f"Ran {', '.join(ran)}." if ran else "Already up to date.")

    def _produce_failed(self, targets: list[str], exc: Exception) -> None:
        _log.error("Could not produce %s", targets, exc_info=exc)
        self._status.setText(f"{type(exc).__name__}: {exc}")
        show_error(str(exc))

    def _produce_finished(self) -> None:
        self._producing = None
        self._refresh()

    # -- the two ways an edit ends ---------------------------------------

    def _revert(self, step: str | None) -> None:
        """Put a step's recorded output back over its layer."""
        if step is None:
            return
        try:
            self._host.revert(step)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"{type(exc).__name__}: {exc}")
            return
        self._status.setText(f"Put {step!r} back to what its run produced.")
        self._refresh()

    def _keep(self, step: str | None) -> None:
        """Keep the hand-made pixels, and stop the step claiming to reproduce.

        Accepting adds a record like everything else: the run stays, and the
        mark after it says the pixels are the user's now.
        """
        if step is None:
            return
        try:
            self._recorder.accept_edit(step)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"{type(exc).__name__}: {exc}")
            return
        self._status.setText(f"Kept the edit to {step!r}.")
        self._refresh()
