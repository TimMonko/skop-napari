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
from psygnal import Signal
from superqt.utils import ensure_main_thread

from skop import OpSpec, Role, Runner, discover

from ._axes import METADATA_KEY
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
        self._progress = ProgressBar(visible=False, min=0, max=0)
        self._results = Container(labels=True, visible=False)

        super().__init__(
            widgets=[
                self._picker,
                self._doc,
                self._inputs_box,
                self._stages_box,
                self._adapt,
                self._mask_view,
                self._z_spacing,
                self._notes,
                self._button,
                self._cancel,
                self._progress,
                self._results,
            ],
            labels=False,
        )

        self._picker.changed.connect(self._select)
        self._mask_view.changed.connect(self._show_z_spacing)
        self._button.changed.connect(self._start)
        self._cancel.changed.connect(self._stop)

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
        for widget in self._inputs.widgets:
            if widget.name in adaptable:
                widget.changed.connect(self._replan)
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
        self._notes.value = self._notes_for(self._inputs, self.spec)
        self._button.enabled = self._inputs.runnable and not self._adapt.problems

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
        # Warnings never block a run. An op fed an axis it did not ask for is
        # the user's call to make; the panel's job is to make sure it is a
        # call they can see themselves making.
        notes.extend(f"Check: {warning}" for warning in self._adapt.warnings)
        return "\n".join(notes)

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
        self._progress.max = 0  # Indeterminate until the op says otherwise.
        # There is no "build started" event, and a first run can sit silent
        # for a while before pixi says anything, so say something ourselves.
        self._progress.label = (
            "Starting workflow"
            if spec.is_workflow
            else f"Preparing environment: {spec.env}"
        )

        args = self._inputs.values()
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
            self._progress.label = f"Preparing environment: {spec.env} ({calls} runs)"

        self._run = OpRun(
            self._runner,
            spec,
            args,
            plans=plans,
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
        self, message: str | None, current: int | None, maximum: int | None
    ) -> None:
        self._show_progress(message, current, maximum)

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
            if maximum:
                self._progress.max = maximum
                self._progress.value = current or 0
            if message:
                self._progress.label = message
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
                self._viewer.add_layer(Layer.create(data, args, layer_type))

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
        self._progress.label = ""
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
