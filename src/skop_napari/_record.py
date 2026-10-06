"""The Record panel: do the work by hand, and keep a record of it.

The Ops panel runs an op and forgets it. This one runs it through a recorder,
so the call is written down -- which op, under what name, with what parameters,
reading which data -- and the log it builds can be rerun, exported or batched
later. The layers the viewer shows are the same layers either way.

What a host is, and what it promises, is in ``_host``: that module imports no
napari and is tested without a viewer. This module is the two things a host
cannot supply for itself, something to draw in and somewhere to press buttons.

A step is named after the op's *function* rather than its output. scikit-ops
names a single unnamed output ``result``, so naming steps after outputs would
put every op into one step and decision 33's identity would be gone. Two ops
that share a function name in different namespaces get the namespace back,
which is also what the picker's labels look like.
"""

from __future__ import annotations

import logging
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import napari.viewer
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
    PushButton,
)
from napari.layers import Layer
from napari.utils.notifications import show_error

from skop import OpSpec, Role, discover, is_op
from skop import spec as spec_of
from skop_core import InProcess, Recorder, SimpleStore
from skop_visual import EDITED, SIGNALS

from ._host import SKOP, Host
from ._panel import _label_for
from ._roles import annotation_for, layer_type_for_role
from ._run import resolve
from ._widget import Inputs, build_inputs

_log = logging.getLogger("skop_napari")

#: The widths the table pads its columns to, and the header that names them.
#: Fixed because a row is one string: magicgui has no table, and the point of
#: the wording is that a column of statuses can be read down.
_STEP_WIDTH = 22
_OP_WIDTH = 34
_HEADER = f"{'step':<{_STEP_WIDTH}}{'op':<{_OP_WIDTH}}state"


def _short_op(op: str) -> str:
    """An op ID, shortened to what tells two of them apart.

    An ID is ``module:function`` and the module is long. What is worth showing
    is the namespace the op lives in -- 'mask: detect' next to 'image: smooth'
    -- so the module's own name is dropped when it repeats the function.
    """
    module, _, function = op.rpartition(":")
    tail = module.rsplit(".", 1)[-1]
    return function if tail == function else f"{tail}:{function}"


def _row_text(node: Any) -> str:
    """One row of the table, as text.

    Text rather than a painted cell: a background colour is a code the reader
    has to learn, and the words are already written. The state is the same
    signal the graph diagram uses, so the two read alike.
    """
    return (
        f"{node.step:<{_STEP_WIDTH}}{_short_op(node.op):<{_OP_WIDTH}}"
        f"{SIGNALS[node.badge]}"
    )


def _fixed_font() -> Any:
    """A monospace font, so the table's columns line up.

    Every failure here is cosmetic and every one of them is better than a
    panel that will not open, so this never raises. The enum is scoped in Qt6
    and unscoped in Qt5, hence the pair of getattr calls rather than one.
    """
    try:
        from qtpy.QtGui import QFontDatabase

        kind = getattr(QFontDatabase, "SystemFont", QFontDatabase)
        return QFontDatabase.systemFont(kind.FixedFont)
    except Exception:  # noqa: BLE001 -- a ragged table beats no panel.
        return None


def _step_name(spec: OpSpec, graph: Any) -> str:
    """What to record a step as: the op's function name.

    The same ops can be run twice under different names one day, which needs a
    widget to type in. There is none yet, and a name derived from the op means
    running the same op again adds a run to the same step -- which is exactly
    what 'do that again' should mean.

    A function name already taken by a *different* op gets its namespace put
    back in front, so the two steps stay two.
    """
    runs = graph.runs_of(spec.function)
    if not runs or runs[-1].op == spec.name:
        return spec.function
    module, _, function = spec.name.rpartition(":")
    return f"{module.rsplit('.', 1)[-1]}_{function}"


class ViewerDisplay:
    """A napari viewer, as a host's display.

    A step's layer is found by the link in its metadata, never by its name: a
    name is a label the user is free to change, and renaming a layer must
    change nothing about what it is.
    """

    def __init__(self, viewer: napari.viewer.Viewer) -> None:
        self._viewer = viewer

    def layers(self) -> list[Any]:
        return list(self._viewer.layers)

    def show(
        self,
        name: str,
        data: Any,
        *,
        step: str,
        role: Role | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Put a step's output where the viewer can see it.

        The layer already drawing this step is given the new data, so running
        a step again does not pile up layers beside the first. A step with no
        layer yet gets one, of the type its role calls for -- a labels output
        lands in a labels layer, not an image one.

        A scalar output has no layer to be. The recorder keeps it; the viewer
        simply does not draw it.
        """
        if not hasattr(data, "shape"):
            _log.info("%s produced %r, which is not an array to draw", step, data)
            return None
        existing = self._layer_for(step)
        if existing is not None:
            existing.data = data
            existing.metadata.update(metadata or {})
            return existing
        args: dict[str, Any] = {"name": name, "metadata": dict(metadata or {})}
        return self._viewer.add_layer(
            Layer.create(data, args, layer_type_for_role(role) or "image")
        )

    def _layer_for(self, step: str) -> Any:
        for layer in self._viewer.layers:
            link = layer.metadata.get(SKOP)
            if link is not None and link.get("step") == step:
                return layer
        return None


class _NoDisplay:
    """A display with nothing in it, for a panel built without a viewer.

    Ten lines here keep every method below free of ``if self._viewer is not
    None``. Nothing reaches a user: napari always supplies a viewer to a dock
    widget whose first parameter is named ``napari_viewer``.
    """

    def layers(self) -> list[Any]:
        return []

    def show(
        self,
        name: str,
        data: Any,
        *,
        step: str,
        role: Role | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return None


class RecordPanel(Container):
    """Runs any op in the collection, and records what it ran."""

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer | None = None,
        specs: list[OpSpec] | None = None,
        registry: dict[str, Any] | None = None,
        root: str | Path | None = None,
    ) -> None:
        # NB: named napari_viewer for the reason the Ops panel gives -- it is
        # the only check napari makes that survives a postponed annotation.
        self._viewer = napari_viewer
        self._specs = list(specs) if specs is not None else list(discover()[0])
        self._registry = (
            dict(registry) if registry is not None else self._resolved_ops()
        )
        # A store is a directory of arrays beside an append-only log, so a
        # session that records anything leaves something to replay. Until
        # there is a way to say where, it goes where throwaway files go, and
        # the panel says so in the tooltip below.
        self._store = SimpleStore(root or tempfile.mkdtemp(prefix="skop-napari-"))
        # The simplest runner, which calls the function here. The panel cannot
        # borrow the Ops panel's, which builds environments with scikit-ops'
        # machinery; the recorder runs ops in this interpreter for now, so an
        # op that wants an environment is refused below rather than run in the
        # wrong one.
        self._recorder = Recorder(self._store, InProcess())
        self._host = Host(
            self._recorder,
            ViewerDisplay(napari_viewer) if napari_viewer is not None else _NoDisplay(),
            self._registry,
        )
        #: Which steps are ticked as targets. Kept here rather than read back
        #: out of the widgets, so rebuilding the table does not lose it.
        self._targets: set[str] = set()
        self._font = _fixed_font()
        self._by_label = {_label_for(spec): spec for spec in self._specs}

        self._picker = ComboBox(
            name="op",
            label="op",
            choices=sorted(self._by_label),
            tooltip="Which op to run, and record.",
        )
        self._doc = Label(value="")
        self._inputs_box = Container(labels=True)
        self._notes = Label(value="")
        self._button = PushButton(text="Run and record")

        self._roots_heading = Label(value="Roots")
        self._roots = Label(value="")
        self._declare = PushButton(text="Declare the viewer's layers")

        self._runs_heading = Label(
            value="Recorded",
            tooltip=f"The log of what was run is written to {self._store.log_path}.",
        )
        self._header = Label(value=_HEADER)
        self._table = Container(labels=False)
        self._produce = PushButton(text="Produce")

        self._status = Label(value="")

        super().__init__(
            widgets=[
                self._picker,
                self._doc,
                self._inputs_box,
                self._notes,
                self._button,
                self._roots_heading,
                self._roots,
                self._declare,
                self._runs_heading,
                self._header,
                self._table,
                self._produce,
                self._status,
            ],
            labels=False,
        )

        self._picker.changed.connect(self._select)
        self._button.changed.connect(self._start)
        self._declare.changed.connect(self._declare_roots)
        self._produce.changed.connect(self._produce_)

        if napari_viewer is not None:
            napari_viewer.layers.events.inserted.connect(self._inserted)
            napari_viewer.layers.events.removed.connect(self._refresh)
        for layer in self._host.display.layers():
            self._watch(layer)
        self._declare_roots()

        if self._by_label:
            self._select()
        else:
            self._doc.value = "No ops found."
            self._button.enabled = False

    def _resolved_ops(self) -> dict[str, Any]:
        """The functions behind the specs, by the ID an op is named by.

        A spec whose module will not import is discovery's business, and is
        already reported there; it is left out here rather than taking the
        whole panel down with it.
        """
        found: dict[str, Any] = {}
        for spec in self._specs:
            try:
                found[spec.name] = resolve(spec)
            except Exception:  # noqa: BLE001 -- see above.
                _log.warning("Could not import %s", spec.name)
        return found

    # -- choosing an op --------------------------------------------------

    @property
    def spec(self) -> OpSpec:
        return self._by_label[self._picker.value]

    def _select(self) -> None:
        """Rebuild the inputs for the newly chosen op."""
        spec = self.spec
        summary = (spec.doc or "").strip().splitlines()
        self._doc.value = summary[0] if summary else ""
        self._inputs: Inputs = build_inputs(spec, annotation_for)
        self._inputs_box.clear()
        self._inputs_box.extend(self._inputs.widgets)

        refusal = self._refusal(spec)
        notes = [refusal] if refusal else []
        if self._inputs.defaulted:
            names = ", ".join(name for name, _ in self._inputs.defaulted)
            notes.append(f"Not adjustable here, using defaults: {names}")
        if self._inputs.blocking:
            names = ", ".join(name for name, _ in self._inputs.blocking)
            notes.append(f"Cannot run: no widget for required input(s) {names}")
        self._notes.value = "\n".join(notes)
        self._button.enabled = refusal is None and self._inputs.runnable

    def _refusal(self, spec: OpSpec) -> str | None:
        """Why the chosen op cannot be recorded, or None when it can.

        The recorder reads scikit-ops' declaration -- ``__skop__``, the marker
        ``@op`` sets -- because that is the contract the log speaks (decision
        44). A function carrying no marker is not an op at all, and is refused
        here in words rather than run and recorded as nothing.
        """
        fn = self._registry.get(spec.name)
        if fn is None:
            return f"{spec.name} is not an op this panel can reach."
        if not is_op(fn):
            return (
                f"{spec.name} is not declared with scikit-ops' @op, so the "
                f"recorder cannot read it (decision 44). Run it from the Ops "
                f"panel instead."
            )
        declared = spec_of(fn)
        if declared.env is not None:
            return (
                f"{spec.name} asks for the environment {declared.env!r}, and "
                f"nothing here can run an op in its own environment yet. "
                f"Running it here instead would ignore the environment it "
                f"asked for."
            )
        return None

    # -- recording a run -------------------------------------------------

    def _start(self) -> None:
        """Run the chosen op once, and write the call down.

        The call happens here, on the GUI thread: the recorder's runner is the
        simplest one, which calls the function. Moving it onto a worker wants
        the scikit-ops runner swapped in behind the recorder, which is the
        next step rather than this one.
        """
        spec = self.spec
        refusal = self._refusal(spec)
        if refusal is not None:
            self._status.value = refusal
            return
        # A layer added since the last run is an input nobody has seen yet.
        self._declare_roots()
        step = _step_name(spec, self._recorder.graph())
        args = self._for_the_recorder(self._inputs.values())
        try:
            self._host.run(step, spec.name, args)
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed.
            _log.exception("Could not record %s", spec.name)
            self._status.value = f"{type(exc).__name__}: {exc}"
            show_error(f"{spec.name}: {exc}")
            return
        self._status.value = f"Recorded {step!r}; its output is a layer."
        self._refresh()

    def _for_the_recorder(self, args: dict[str, Any]) -> dict[str, Any]:
        """Hand the host the layers, not the arrays a widget happened to hold.

        A magicgui layer combo hands over a layer or the array behind it,
        depending on how the parameter was annotated, and the host resolves a
        handle from a *layer* -- which is the only thing that knows whether
        what it holds is still what was recorded. So an array that came out of
        a layer is sent back to it, found by identity, which is the same test
        the store makes. Anything else -- a number, a string, an array from
        somewhere else -- passes through untouched.
        """
        return {name: self._owner(value) or value for name, value in args.items()}

    def _owner(self, value: Any) -> Any:
        for layer in self._host.display.layers():
            if layer.data is value:
                return layer
        return None

    # -- roots -----------------------------------------------------------

    def _declare_roots(self) -> None:
        """Take every layer the host has not seen as a root.

        Idempotent, and safe to press twice: a layer already carrying its link
        is left alone, which is also what makes a duplicate of a layer a view
        of the same step rather than a new input.
        """
        declared = self._host.declare_roots()
        if declared:
            _log.info("Declared roots: %s", ", ".join(declared))
        self._refresh()

    def _roots_text(self) -> str:
        names = [
            layer.name
            for layer in self._host.display.layers()
            if (layer.metadata.get(SKOP) or {}).get("step", "missing") is None
        ]
        if not names:
            return "Nothing yet. An op needs a layer to read."
        return ", ".join(names)

    # -- what was recorded -----------------------------------------------

    def _refresh(self) -> None:
        """Rebuild the table from the graph, which is the only account of it.

        Read back from the log rather than remembered: a panel that kept its
        own idea of what had been run is a second source of truth, and the one
        that goes wrong.
        """
        view = self._host.view()
        steps = {node.step for node in view.nodes}
        self._targets &= steps

        self._header.native.setFont(self._font) if self._font else None
        self._table.clear()
        for node in view.nodes:
            row = Container(
                widgets=[
                    Label(value=_row_text(node)),
                    CheckBox(value=node.step in self._targets, tooltip="Produce this"),
                ],
                labels=False,
            )
            if self._font:
                for widget in row:
                    widget.native.setFont(self._font)
            row[1].changed.connect(partial(self._tickle, node.step))
            if EDITED in node.flags:
                # Only an edited step needs these, so they are only built for
                # one: a button that does nothing is worse than no button.
                row.extend(self._edit_buttons(node.step))
            self._table.append(row)

        self._roots.value = self._roots_text()
        self._produce.enabled = bool(self._targets)

    def _edit_buttons(self, step: str) -> list[Any]:
        revert = PushButton(text="Revert")
        revert.changed.connect(partial(self._revert, step))
        keep = PushButton(text="Keep the edit")
        keep.changed.connect(partial(self._keep, step))
        return [revert, keep]

    def _tickle(self, step: str, ticked: bool) -> None:
        if ticked:
            self._targets.add(step)
        else:
            self._targets.discard(step)
        self._produce.enabled = bool(self._targets)
        self._status.value = self._targets_line()

    def _targets_line(self) -> str:
        if not self._targets:
            return "Tick the steps to bring up to date, then Produce."
        return f"Targets: {', '.join(sorted(self._targets))}"

    def _produce_(self) -> None:
        """Bring the ticked steps up to date, and draw what moved.

        Produce runs the backward closure of the targets and nothing else
        (decision 4), so a step whose inputs have not moved is left alone and
        costs nothing.
        """
        if not self._targets:
            self._status.value = self._targets_line()
            return
        try:
            ran = self._host.produce(sorted(self._targets))
        except Exception as exc:  # noqa: BLE001 -- reported, not swallowed.
            _log.exception("Could not produce %s", sorted(self._targets))
            self._status.value = f"{type(exc).__name__}: {exc}"
            show_error(str(exc))
            return
        for step in ran:
            try:
                self._host.show(step)
            except Exception:  # noqa: BLE001 -- a scalar output has no layer.
                _log.info("Nothing to draw for %r", step)
        self._status.value = f"Ran {', '.join(ran)}." if ran else "Already up to date."
        self._refresh()

    # -- the two ways an edit ends ---------------------------------------

    def _revert(self, step: str) -> None:
        """Put a step's recorded output back over its layer."""
        try:
            self._host.revert(step)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"{type(exc).__name__}: {exc}"
            return
        self._status.value = f"Put {step!r} back to what its run produced."
        self._refresh()

    def _keep(self, step: str) -> None:
        """Keep the hand-made pixels, and stop the step claiming to reproduce.

        Accepting adds a record like everything else: the run stays, and the
        mark after it says the pixels are the user's now.
        """
        try:
            self._recorder.accept_edit(step)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"{type(exc).__name__}: {exc}"
            return
        self._status.value = f"Kept the edit to {step!r}."
        self._refresh()

    # -- keeping up with the viewer --------------------------------------

    def _inserted(self, event: Any) -> None:
        """A layer appeared: watch it, and take it as a root.

        Every layer the viewer holds that this host has not seen is an input,
        which is what makes "open an image, then run an op" work with no
        declaration step in between. A layer the host drew itself already
        carries its link and is left alone.
        """
        self._watch(event.value)
        self._declare_roots()

    def _watch(self, layer: Any) -> None:
        """Listen for a layer's data changing by hand."""
        layer.events.data.connect(self._changed)

    def _changed(self, event: Any) -> None:
        """Note a hand edit, or a root moving, and show it in the table."""
        try:
            mattered = self._host.data_changed(event.source)
        except Exception:  # noqa: BLE001 -- an event handler must not raise.
            _log.exception("Could not read the change to %r", event.source)
            return
        if mattered:
            self._status.value = (
                "That layer is edited now: revert it, or keep the edit."
            )
            self._refresh()
