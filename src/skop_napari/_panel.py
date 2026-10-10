"""The Ops panel: pick an op, fill in its inputs, run it.

One widget for the whole collection, rather than one contribution per op.
npe2 wants a static manifest, and skop discovers its ops by importing them,
so a per-op contribution would mean generating the manifest at build time and
losing the ability to drop a new op into the collection and see it appear.
The napari team is working on dynamic registration; when it lands, this file
is what gets split up.

There are two widgets, not one, and the second is what gets skop-napari a
submenu: napari builds ``Plugins > scikit-ops > ...`` as soon as a plugin
contributes more than one widget, and leaves a single one flat. Ops and
workflows are the split worth having -- a workflow is an op, so one class
serves both, differing only in which half of the collection it offers.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import napari.viewer
from magicgui.widgets import (
    ComboBox,
    Container,
    FloatSpinBox,
    Label,
    LineEdit,
    ProgressBar,
    PushButton,
)
from napari.layers import Layer
from napari.utils.notifications import notification_manager, show_error
from opspec.op import OpSpec, Role
from opspec.tiling import TilePlan, parse_bytes
from psygnal import Signal
from superqt.utils import ensure_main_thread

from skop import Runner, discover
from skop._tiling import DEFAULT_FRACTION, Tiler, default_budget

from ._axes import METADATA_KEY, layer_for
from ._choices import is_workflow_plumbing, stages_for
from ._plans import Adaptations
from ._roles import (
    DEFAULT_MASK_VIEW,
    DEFAULT_Z_SPACING,
    MASK_VIEWS,
    annotation_for,
    is_stack_view,
    layer_args_for,
    layer_type_for,
    value_for,
)
from ._run import OpRun, outputs_of, resolve
from ._widget import Inputs, build_inputs

# Output names too generic to serve as layer names on their own. "boxes" is
# here for the same reason "labels" is: every detector produces some, they are
# meant to be interchangeable, and two in one session would otherwise both be
# called "boxes" with only a napari-supplied counter to tell them apart.
_GENERIC = frozenset(
    {
        "boxes",
        "cells",
        "image",
        "labels",
        "mask",
        "nuclei",
        "out",
        "output",
        "points",
        "result",
    }
)

_log = logging.getLogger("skop_napari")

# A verbose-log line from the build tool, e.g. "DEBUG pixi_config: ...".
_LOG_RECORD = re.compile(r"^\s*(TRACE|DEBUG|INFO|WARN|WARNING|ERROR)\b")


def _label_for(spec: OpSpec) -> str:
    """A human-facing name for an op, e.g. 'segment: stardist2d'.

    The ``workflows.`` prefix is stripped rather than shown. Workflows live in
    their own tree so that a person reading the source can see what has been
    assembled, but they have their own panel here, where saying "workflows" on
    every row is noise. What survives is the domain underneath it, which is
    what the label is for: 'mask: detect_then_mask' sits next to the mask ops
    it belongs with.
    """
    path = spec.module.removeprefix("skop.ops.").removeprefix("workflows.")
    namespace = path.split(".")[0]
    return (
        spec.function if namespace == spec.function else f"{namespace}: {spec.function}"
    )


def _where(spec: OpSpec) -> str:
    """Where an op will run, for the notes line."""
    if spec.is_workflow:
        return "Runs on the host; each stage runs in its own environment."
    return f"Environment: {spec.env}"


def _describe(value: Any) -> str:
    """A short description of an argument, for the log."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"{type(value).__name__}{tuple(shape)} {getattr(value, 'dtype', '')}"
    return repr(value)


def _follow(layer: Layer, source: Layer | None) -> None:
    """Put an output where its input is: the same scale and offset.

    An op works in the input's pixels, so its boxes and labels come back in
    them. A layer shown at 8x scale -- level 3 of a pyramid, say -- would
    otherwise get its outputs drawn at 1x, squeezed into the top-left corner.
    An output with more dimensions than its input (a stack of masks) is
    unscaled along the extra, leading ones.
    """
    if source is None:
        return
    extra = max(0, layer.ndim - source.ndim)
    scale = tuple(source.scale)[-layer.ndim :]
    translate = tuple(source.translate)[-layer.ndim :]
    layer.scale = (1.0,) * extra + scale
    layer.translate = (0.0,) * extra + translate


def _zarr_out_for(data: Any, plan: TilePlan) -> Any:
    """Where a tiled run on a zarr puts its result: another zarr.

    A run that has to be tiled has a result too big for memory, and an input
    that is a zarr says zarr is what this user works in. So the result is
    one too, in a temporary folder, chunked the way it is tiled, and shown
    as lazily as the input was. Made once the result's dtype is known, which
    is why this hands the runner a function rather than an array.

    Any other input returns None, and the runner's own default stands.
    """
    if type(data).__module__.split(".")[0] != "zarr":
        return None
    import zarr

    path = Path(tempfile.mkdtemp(prefix="skop-tiled-")) / "result.zarr"

    def make(shape: tuple[int, ...], dtype: Any) -> Any:
        chunks = tuple(min(c, n) for c, n in zip(plan.tile_shape, shape))
        return zarr.open_array(
            str(path), mode="w", shape=shape, dtype=dtype, chunks=chunks
        )

    return make


#: How many tiles a run shows a bar for at once. A big image may be
#: hundreds of tiles; when this many are full, they are cleared.
TILE_BARS = 4

#: The overlap a workflow's plan is estimated with when the field is empty.
#: The op's own overlap may come from a PSF the workflow makes partway
#: through, after the panel has planned. The run uses the op's own, from the
#: real PSF. 10 is what clij2-fft and tnia-python use.
WORKFLOW_OVERLAP = 10


def _tile_line(plan: TilePlan) -> str:
    """The plan in one line: how many tiles, and what each needs."""
    line = f"{plan.summary}; needs {_size(plan.peak)} of {_size(plan.budget)}"
    return line + " -- over the budget" if plan.peak > plan.budget else line


def _numbers(text: str) -> tuple[int, ...] | None:
    """Whole numbers typed into a field, "64 256 256" or "64x256x256"; None if empty."""
    parts = text.replace("x", " ").replace(",", " ").split()
    if not parts:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        raise ValueError(f"not whole numbers: {text.strip()!r}") from None


def _budget_from(text: str) -> float | int | None:
    """A memory budget as typed: empty, a share of what is free, or a size."""
    if not text:
        return None
    try:
        share = float(text)
    except ValueError:
        return parse_bytes(text)
    if 0 < share <= 1:
        return share
    raise ValueError(f"a share of free memory is between 0 and 1, not {share}")


def _size(n: int) -> str:
    """Bytes as a size someone would type: 19.3G, 850M."""
    for unit, scale in (("T", 1024**4), ("G", 1024**3), ("M", 1024**2)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return f"{n // 1024}K"


def _layer_name(spec: OpSpec, output: str) -> str:
    if output.lower() in _GENERIC:
        return f"{output} [{spec.function}]"
    return output


class OpsPanel(Container):
    """Runs any op in the collection, against the layers in the viewer."""

    #: Emitted when a run ends, however it ended -- completed, failed or
    #: canceled. Ops finish on a worker thread, so this is how anything
    #: driving the panel from outside knows the outputs have landed.
    finished = Signal()

    #: Which half of the collection this panel offers. Workflows are ops and
    #: run through the same machinery, so the split is presentational: a
    #: dropdown of sixty leaf ops is a bad place to find the four things
    #: someone actually assembled, and vice versa.
    shows_workflows = False

    #: What the picker says when the filter leaves nothing.
    empty_message = "No ops found."

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer | None = None,
        runner: Runner | None = None,
    ) -> None:
        # NB: the parameter must be *named* napari_viewer. That is the first
        # thing napari checks when instantiating a dock widget, and the only
        # check that works here: its other route compares the annotation
        # against the Viewer type, and `from __future__ import annotations`
        # turns every annotation in this module into a string. Get this wrong
        # and napari constructs the panel with no viewer at all, leaving
        # nowhere to put an op's output.
        self._viewer = napari_viewer
        # One Runner for the session: workers stay warm between invocations,
        # which is most of why running an op a second time is fast.
        #
        # napari constructs this widget itself, so there is no call site to
        # pass debug=True at; SKOP_NAPARI_DEBUG is the way in. It echoes the
        # worker's own stdout and stderr, which is the only view of what the
        # op is doing inside its own process.
        self._runner = (
            runner
            if runner is not None
            else Runner(debug=bool(os.environ.get("SKOP_NAPARI_DEBUG")))
        )
        self._run: OpRun | None = None
        # How the chosen op would be tiled under the memory budget, if at all,
        # and what is wrong with the budget, if anything.
        self._tile_plan: TilePlan | None = None
        self._tile_problem: str | None = None

        self._specs, self._failures = discover()
        self._by_label = {
            _label_for(s): s
            for s in self._specs
            if s.is_workflow is self.shows_workflows
        }

        self._picker = ComboBox(
            name="op",
            label="op",
            choices=sorted(self._by_label),
            tooltip="Which op to run.",
        )
        self._doc = Label(value="")
        self._inputs_box = Container(labels=True)
        # Workflow stages live in their own container, with labels off. A
        # Container unifies its label column across every child, so a stage
        # sharing _inputs_box would be indented by the widest parameter name
        # in the panel -- 70px of a 425px dock, for a label its own heading
        # already shows. Empty for an ordinary op.
        self._stages_box = Container(labels=False)
        self._adapt = Adaptations()
        # A memory budget, offered only for ops that say how they can be cut
        # up (@op(tile=..., peak_memory=...)). Over it, the op runs tile by
        # tile; under it, or with the box empty, it runs whole as always.
        self._memory = LineEdit(
            name="memory",
            label="memory budget",
            value="",
            tooltip=(
                "Run the op in tiles that each fit this much memory: a size, "
                "500M or 2G, or a share of what is free, 0.5. Empty: 85% of "
                "the memory free right now, which only tiles an input too big "
                "for it. 'off': always run whole."
            ),
        )
        # A Tiler's two overrides (0017 tiling.md, "The Tiler"). Empty, each
        # shows what would be used, and follows the budget as it changes.
        self._tile_size = LineEdit(
            name="tile_size",
            label="tile size",
            value="",
            tooltip=(
                "The size of each tile, one number per axis, e.g. 64 256 256. "
                "Empty: the largest that fits the budget."
            ),
        )
        self._overlap = LineEdit(
            name="overlap",
            label="overlap",
            value="",
            tooltip=(
                "How far each tile reaches into its neighbours, in pixels: one "
                "number, or one per axis. Empty: the op's own."
            ),
        )
        # The panel is built with labels=False; this box draws the labels.
        self._memory_box = Container(
            labels=True,
            widgets=[self._memory, self._tile_size, self._overlap],
            visible=False,
        )
        self._tiles = Label(value="", visible=False)
        # How to show a mask collection. A display choice rather than an op
        # parameter: nothing about it reaches the model, and re-showing a
        # result a different way should not mean running SAM again. Hidden
        # unless the chosen op actually produces masks.
        self._mask_view = ComboBox(
            name="mask_view",
            label="show masks as",
            choices=list(MASK_VIEWS),
            value=DEFAULT_MASK_VIEW,
            visible=False,
            tooltip=(
                "Masks may overlap, so no single label image can show all of "
                "them. Project to 2-D and pick who wins a contested pixel, or "
                "keep the 3-D stack, which loses nothing."
            ),
        )
        # Only meaningful for the 3-D stack, so it follows the chooser.
        self._z_spacing = FloatSpinBox(
            name="z_spacing",
            label="stack spacing",
            value=DEFAULT_Z_SPACING,
            min=0.1,
            max=1000.0,
            step=1.0,
            visible=False,
            tooltip=(
                "How far apart to draw the planes of the 3-D stack, relative "
                "to a pixel. The axis is an object index rather than a depth, "
                "so there is no true value -- at 1 a handful of masks is a "
                "pancake and rotating it shows nothing."
            ),
        )
        self._notes = Label(value="")
        self._button = PushButton(text="Run")
        self._cancel = PushButton(text="Cancel", visible=False)
        # The op's progress messages. Not the bar's own label: this panel is
        # built with labels=False, so magicgui never draws that.
        self._status = Label(value="", visible=False)
        self._progress = ProgressBar(visible=False, min=0, max=0)
        # A tiled run's last few tiles, a bar each; the main bar counts tiles.
        self._tile_bars = Container(labels=True, visible=False)
        self._tile: tuple[int, int] | None = None
        self._results = Container(labels=True, visible=False)

        super().__init__(
            widgets=[
                self._picker,
                self._doc,
                self._inputs_box,
                self._stages_box,
                self._adapt,
                self._memory_box,
                self._tiles,
                self._mask_view,
                self._z_spacing,
                self._notes,
                self._button,
                self._cancel,
                self._status,
                self._progress,
                self._tile_bars,
                self._results,
            ],
            labels=False,
        )

        self._picker.changed.connect(self._select)
        self._mask_view.changed.connect(self._show_z_spacing)
        self._button.changed.connect(self._start)
        self._cancel.changed.connect(self._stop)
        self._memory.changed.connect(self._replan)
        self._tile_size.changed.connect(self._replan)
        self._overlap.changed.connect(self._replan)

        # "Current position" means the position right now, not the one that
        # happened to be showing when the op was picked. Without this the
        # panel says z=33 while the viewer sits at z=20, and Run believes the
        # panel.
        if self._viewer is not None:
            self._viewer.dims.events.current_step.connect(self._moved)

        # Building an environment happens inside run(), on the worker thread,
        # and is the slowest part of a first run by a wide margin. These
        # callbacks are the only way to show anything while it happens.
        self._runner.subscribe_build_progress(
            ensure_main_thread(self._on_build_progress)
        )
        self._runner.subscribe_build_output(ensure_main_thread(self._on_build_text))
        # NB: not an error channel despite the name -- this is the build
        # tool's stderr, where pixi writes all its status, success included.
        self._runner.subscribe_build_error(ensure_main_thread(self._on_build_text))

        if self._by_label:
            self._select()
        else:
            self._doc.value = self.empty_message
            self._button.enabled = False

    # -- op selection ----------------------------------------------------

    @property
    def spec(self) -> OpSpec:
        return self._by_label[self._picker.value]

    def _select(self) -> None:
        """Rebuild the input widgets for the newly chosen op."""
        spec = self.spec
        summary = (spec.doc or "").strip().splitlines()
        self._doc.value = summary[0] if summary else ""

        # A workflow's choosers are drawn as groups, so its own parameters
        # skip them here and the groups are appended after.
        self._inputs: Inputs = build_inputs(
            spec, annotation_for, value_for, skip=is_workflow_plumbing
        )
        self._inputs.extra = stages_for(spec, annotation_for, value_for)
        self._inputs_box.clear()
        self._inputs_box.extend(self._inputs.widgets)
        self._stages_box.clear()
        self._stages_box.extend(self._inputs.extra)

        self._mask_view.visible = self._makes_masks()
        self._show_z_spacing()

        adaptable = self._adapt.rebuild(spec)
        # Which array is selected decides what can be done with it, so the
        # adaptation rows follow the layer combos rather than being computed
        # once. The rows themselves also re-plan when their axes are edited.
        # The tile line follows the input being tiled, and whichever
        # parameter sizes the overlap.
        tiled = self._tiled()
        watched = set(adaptable) | set(tiled[0].tile if tiled else ())
        if spec.overlap and spec.overlap.param:
            watched.add(spec.overlap.param)
        for widget in self._inputs.widgets:
            if widget.name in watched:
                widget.changed.connect(self._replan)
        # A workflow's choosers decide which op is tiled, and on which device.
        for stage in self._inputs.extra:
            stage.changed.connect(self._replan)
        self._adapt.watch(self._replan)
        self._replan()

        self._results.clear()
        self._results.visible = False

    def _makes_masks(self) -> bool:
        return any(output.role is Role.masks for output in self.spec.outputs)

    def _show_z_spacing(self) -> None:
        """Offer the stack spacing only when a stack is what will be made.

        Asked of the spec rather than of ``self._mask_view.visible``: that
        reads Qt's idea of visibility, which is False for every widget of an
        unshown parent, so gating on it would make this depend on whether the
        panel happened to be painted yet.
        """
        self._z_spacing.visible = self._makes_masks() and is_stack_view(
            self._mask_view.value
        )

    def _replan(self) -> None:
        """Re-resolve axes and re-plan, then say whether the op can run."""
        self._adapt.refresh(resolve(self.spec), self._inputs.values(), self._viewer)
        tiled = self._tiled()
        self._memory_box.visible = tiled is not None
        if tiled:
            device = tiled[0].peak_memory.device
            free = "GPU memory" if device == "gpu" else "free memory"
            budget = default_budget(DEFAULT_FRACTION, device)
            self._memory.native.setPlaceholderText(
                f"auto: {_size(budget)} ({DEFAULT_FRACTION:.0%} of {free})"
                if budget is not None
                else f"auto: {free} unknown, so not tiled"
            )
        self._tile_plan, self._tile_problem = self._plan_tiles()
        plan = self._tile_plan
        self._tiles.value = _tile_line(plan) if plan else ""
        if plan and self.spec.is_workflow:
            self._tiles.value += " (estimated: the PSF is made when it runs)"
        self._tiles.visible = plan is not None
        if plan is not None:
            # What would be used, in the fields the user has left empty.
            self._tile_size.native.setPlaceholderText(
                "auto: " + " x ".join(str(n) for n in plan.tile_shape)
            )
            reach = plan.overlap
            self._overlap.native.setPlaceholderText(
                "auto: "
                + (
                    " x ".join(str(n) for n in reach)
                    if isinstance(reach, tuple)
                    else str(reach)
                )
            )
        self._notes.value = self._notes_for(self._inputs, self.spec)
        self._button.enabled = (
            self._inputs.runnable
            and not self._adapt.problems
            and not self._tile_problem
        )

    def _moved(self) -> None:
        """Re-plan when the viewer's sliders move, if that changes anything.

        Dragging a slider fires this once per step, and re-planning rebuilds
        every combo in the row, so it is worth doing only when a plan actually
        reads the position -- which is exactly when some axis is set to
        "current position". Every other plan is the same plan wherever the
        sliders are.
        """
        if not any(plan.select for plan in self._adapt.plans.values()):
            return
        try:
            self._replan()
        except RuntimeError:
            # The panel was closed but the viewer outlived it, so this event
            # is still arriving at widgets whose Qt objects are gone.
            self._viewer.dims.events.current_step.disconnect(self._moved)

    def _notes_for(self, inputs: Inputs, spec: OpSpec) -> str:
        notes = [_where(spec)]
        if inputs.defaulted:
            names = ", ".join(name for name, _ in inputs.defaulted)
            notes.append(f"Not editable here, using defaults: {names}")
        if inputs.blocking:
            names = ", ".join(name for name, _ in inputs.blocking)
            notes.append(f"Cannot run: no widget for required input(s) {names}")
        notes.extend(note for stage in inputs.extra for note in stage.notes())
        notes.extend(f"Cannot run: {problem}" for problem in self._adapt.problems)
        if self._tile_problem:
            notes.append(f"Cannot run: memory budget: {self._tile_problem}")
        # Warnings never block a run. An op fed an axis it did not ask for is
        # the user's call to make; the panel's job is to make sure it is a
        # call they can see themselves making.
        notes.extend(f"Check: {warning}" for warning in self._adapt.warnings)
        return "\n".join(notes)

    def _tiler(self, estimate: bool = False) -> tuple[Tiler | None, str | None]:
        """The Tiler the three fields describe, or what is wrong with them.

        With *estimate*, for showing a workflow's plan before it runs: an
        empty overlap is WORKFLOW_OVERLAP.
        """
        text = self._memory.value.strip()
        if text.lower() == "off":
            return Tiler.off(), None
        try:
            memory = _budget_from(text)
            tile_size = _numbers(self._tile_size.value)
            overlap = _numbers(self._overlap.value)
        except ValueError as exc:
            return None, str(exc)
        if overlap is not None and len(overlap) == 1:
            overlap = overlap[0]
        if overlap is None and estimate and self.spec.is_workflow:
            overlap = WORKFLOW_OVERLAP
        return Tiler(memory=memory, overlap=overlap, tile_shape=tile_size), None

    def _tiled(self) -> tuple[OpSpec, dict[str, Any]] | None:
        """The op that may be tiled, and its arguments as the panel has them.

        The chosen op, or for a workflow, the op one of its choosers picked
        that declares tiling hints. A workflow passes its own inputs to that
        op by name, so the workflow's image is the one tiled.
        """
        spec = self.spec
        own = self._inputs.values()
        if spec.tile and spec.peak_memory:
            values = {p.name: p.default for p in spec.params if not p.required}
            return spec, {**values, **own}
        for stage in self._inputs.extra:
            sub = stage.spec
            if sub.tile and sub.peak_memory:
                values = {p.name: p.default for p in sub.params if not p.required}
                values.update(stage.values())
                values.update({name: own[name] for name in sub.tile if name in own})
                return sub, values
        return None

    def _plan_tiles(self) -> tuple[TilePlan | None, str | None]:
        """How the chosen op would be tiled, or why it can't be.

        Asked of the same Tiler the run is given, so the panel says what the
        run will do. Shown even when the whole input is one tile, so the user
        sees what it needs. None when the op cannot be tiled, tiling is off,
        or the budget for its device is not known.
        """
        tiled = self._tiled()
        if tiled is None:
            return None, None
        spec, values = tiled
        tiler, problem = self._tiler(estimate=True)
        if problem:
            return None, problem
        data = values.get(spec.tile[0])
        if not (hasattr(data, "shape") and hasattr(data, "dtype")):
            return None, None  # No layer chosen yet, or a pyramid.
        for name, given in (
            ("tile size", tiler.tile_shape),
            ("overlap", tiler.overlap),
        ):
            if isinstance(given, tuple) and len(given) != data.ndim:
                return None, f"{name} needs {data.ndim} numbers, one per axis"
        try:
            return tiler.plan(spec, values), None
        except ValueError as exc:
            return None, str(exc)

    # -- running ---------------------------------------------------------

    def _start(self) -> None:
        if self._run is not None:
            return

        spec = self.spec
        self._results.clear()
        self._results.visible = False
        self._button.enabled = False
        self._cancel.visible = True
        self._progress.visible = True
        self._status.visible = True
        self._progress.max = 0  # Indeterminate until the op says otherwise.
        # There is no "build started" event, and a first run can sit silent
        # for a while before pixi says anything, so say something ourselves.
        self._status.value = (
            "Starting workflow"
            if spec.is_workflow
            else f"Preparing environment: {spec.env}"
        )

        args = self._inputs.values()
        # The first input that is a layer; outputs are drawn where it is.
        self._source_layer = next(
            (
                layer
                for w in self._inputs.widgets
                if (layer := layer_for(self._viewer, w.value)) is not None
            ),
            None,
        )
        # Frozen for the duration of the run: the user is free to change the
        # selection while it works, and the outputs still belong to the plan
        # that produced them.
        plans = self._adapt.plans
        self._output_axes = self._adapt.output_axes
        _log.info(
            "Running %s in env %r with %s%s",
            spec.name,
            spec.env,
            {k: _describe(v) for k, v in args.items()},
            "".join(f"\n  {name}: {plan.summary}" for name, plan in plans.items()),
        )
        if any(plan.calls > 1 for plan in plans.values()):
            calls = max(plan.calls for plan in plans.values())
            self._status.value = f"Preparing environment: {spec.env} ({calls} runs)"

        tiler = self._tiler()[0] if self._tiled() else None
        out = None
        if self._tile_plan and self._tile_plan.calls > 1:
            _log.info("Tiling: %s", _tile_line(self._tile_plan))
            # A workflow returns several results; only an op's one goes to out.
            if not spec.is_workflow:
                out = _zarr_out_for(args.get(spec.tile[0]), self._tile_plan)

        self._run = OpRun(
            self._runner,
            spec,
            args,
            plans=plans,
            tiler=tiler,
            out=out,
            on_progress=self._on_progress,
            on_done=self._on_done,
            on_error=self._on_error,
            on_finish=self._on_finish,
        )
        self._run.start()

    def _stop(self) -> None:
        if self._run is not None:
            self._cancel.enabled = False
            self._cancel.text = "Canceling..."
            self._run.cancel()

    def _on_progress(
        self,
        message: str | None,
        current: int | None,
        maximum: int | None,
        tile: tuple[int, int] | None = None,
    ) -> None:
        if tile is None:
            self._show_progress(message, current, maximum)
            return
        try:
            self._next_tile(*tile)
        except RuntimeError:
            if self._run is not None:  # closed mid-run; see _show_progress
                self._run.detach()

    def _next_tile(self, number: int, count: int) -> None:
        """Start tile *number* of *count*: count it, and give it a bar."""
        self._tile = (number, count)
        self._progress.max = count
        self._progress.value = number - 1
        self._status.value = f"Tile {number} of {count}"
        bars = self._tile_bars
        if len(bars):
            bars[-1].value = bars[-1].max  # tiles run one at a time
        if len(bars) >= TILE_BARS:
            bars.clear()
        bars.append(ProgressBar(min=0, max=0, label=f"tile {number} of {count}"))
        bars.visible = True

    # -- environment building --------------------------------------------

    def _on_build_progress(self, title: str, current: int, maximum: int) -> None:
        """Report determinate build progress, e.g. downloading pixi itself."""
        self._show_progress(title, current, maximum)

    def _on_build_text(self, text: str) -> None:
        """Report a chunk of the build tool's output.

        Everything goes to the log, where the detail is usable. Only the
        occasional human-facing line reaches the progress bar: Appose runs
        pixi under -vv to drive PixiInstallMonitor, so most of this stream is
        pixi's own debug log rather than anything worth showing.
        """
        _log.info("%s", text.rstrip())
        status = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not _LOG_RECORD.match(line)
        ]
        if status:
            self._show_progress(status[-1], None, None)

    def _show_progress(
        self, message: str | None, current: int | None, maximum: int | None
    ) -> None:
        try:
            # In a tiled run, the op's own progress is the current tile's.
            bar = self._tile_bars[-1] if self._tile else self._progress
            if maximum:
                bar.max = maximum
                bar.value = current or 0
            if message:
                where = (
                    f"Tile {self._tile[0]} of {self._tile[1]}: " if self._tile else ""
                )
                self._status.value = where + message
        except RuntimeError:
            # The panel was closed while its op was still running, so the Qt
            # objects behind these widgets are gone. The op carries on in its
            # own process; there is simply nowhere left to report it.
            if self._run is not None:
                self._run.detach()

    def _on_done(self, result: Any) -> None:
        spec = self.spec
        values = outputs_of(spec, result)
        scalars: list[tuple[str, Any]] = []
        dropped: list[str] = []

        for output in spec.outputs:
            value = values.get(output.name)
            if value is None:
                _log.info("Op %s returned no %s", spec.name, output.name)
                continue
            layer_type = layer_type_for(output)
            if layer_type is None:
                scalars.append((output.name, value))
            elif self._viewer is None:
                dropped.append(output.name)
            else:
                name = _layer_name(spec, output.name)
                _log.info("Adding %s layer %r from %s", layer_type, name, spec.name)
                # Two separate jobs, in order. layer_args_for adapts the data
                # to the layout napari wants; _layer_args then says what to
                # call the layer and stamps its axes. Asking _layer_args about
                # the *adapted* data rather than the raw value is deliberate:
                # its axis check compares against ndim, so boxes reshaped out
                # of image space stop matching the image's axis count and are
                # correctly left unlabelled -- a Shapes layer's dimensions are
                # coordinates, not axes.
                data, extra = layer_args_for(
                    output, value, self._mask_view.value, self._z_spacing.value
                )
                args = self._layer_args(name, data)
                args.update(extra)
                layer = Layer.create(data, args, layer_type)
                _follow(layer, getattr(self, "_source_layer", None))
                self._viewer.add_layer(layer)

        self._show_scalars(scalars)

        if dropped:
            # Never fail quietly here. Without a viewer an op appears to run
            # perfectly -- progress runs to completion, no error -- and its
            # output simply evaporates, which is indistinguishable from the
            # op being broken.
            message = (
                f"{spec.name} produced {', '.join(dropped)}, but this panel has "
                f"no viewer to put them in. It was constructed without one; "
                f"napari supplies a viewer only to a parameter named "
                f"'napari_viewer'."
            )
            _log.error("%s", message)
            show_error(message)

    def _layer_args(self, name: str, value: Any) -> dict[str, Any]:
        """Layer keywords for one output, carrying its axes if we know them.

        Stamping the result closes the loop: the next op run against this
        layer reads its axes off rung one instead of guessing again, and the
        viewer's sliders are labelled 'z' rather than '-3'.

        Only a complete answer is stamped. Half of one would leave napari
        showing a blank slider label, and would come back on the next run as a
        declaration nobody made.
        """
        args: dict[str, Any] = {"name": name}
        axes = getattr(self, "_output_axes", ())
        if axes and all(axes) and len(axes) == getattr(value, "ndim", -1):
            args["metadata"] = {METADATA_KEY: tuple(axes)}
            args["axis_labels"] = tuple(axes)
        return args

    def _show_scalars(self, scalars: list[tuple[str, Any]]) -> None:
        """Display outputs that are not layers -- counts, measurements."""
        self._results.clear()
        for name, value in scalars:
            field = LineEdit(name=name, label=name, value=str(value))
            field.enabled = False
            self._results.append(field)
        self._results.visible = bool(scalars)

    def _on_error(self, exc: Exception) -> None:
        """Report a failed op through napari's own error machinery.

        A one-line LineEdit in the panel could not show a traceback, and the
        interesting part of an op failure is the traceback -- it comes from
        another process, running another interpreter, and says what actually
        went wrong in there. napari's notification carries it, keeps it in
        the notification history, and looks like every other napari error.
        """
        _log.error("Op %s failed", self.spec.name, exc_info=exc)
        notification_manager.receive_error(type(exc), exc, exc.__traceback__)

    def _on_finish(self) -> None:
        self._run = None
        self._button.enabled = self._inputs.runnable
        self._cancel.visible = False
        self._cancel.enabled = True
        self._cancel.text = "Cancel"
        self._progress.visible = False
        self._tile_bars.clear()
        self._tile_bars.visible = False
        self._tile = None
        self._status.visible = False
        self._status.value = ""
        self.finished.emit()


class WorkflowsPanel(OpsPanel):
    """Runs any workflow in the collection.

    The whole of the difference is which ops it lists. A workflow is an op --
    it discovers, specs, renders and runs exactly like one -- so a second
    class here would be a second protocol to no purpose. What it buys is the
    submenu: napari nests a plugin's widgets under its own name only when the
    plugin contributes more than one, so this is also what turns
    ``Plugins > scikit-ops: Ops`` into ``Plugins > scikit-ops > ...``.
    """

    shows_workflows = True
    empty_message = "No workflows found."
