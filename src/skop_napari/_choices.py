"""Rendering a workflow: a combo box per stage, with that stage's own inputs.

A workflow is an op whose parameters include *other ops*. Two of its
parameters go together -- a ``Choices`` chooser naming what may be run, and
the ``ParamsFor`` dict holding the chosen op's settings -- so they are drawn
as one thing: a combo box with the chosen op's widgets underneath it,
rebuilt whenever the combo changes.

Which is nearly free, because the widgets under a chooser are built by the
same ``build_inputs`` that builds the panel's own. An op is an op; a workflow
stage is drawn exactly like a top-level op, one level in.

The parameters a workflow supplies itself are not drawn. The workflow says
which those are through ``ParamsFor(..., binds=...)`` -- both stages of
detect_then_mask take the image, and asking for it twice would be the panel
misrepresenting what is about to happen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from magicgui.widgets import ComboBox, Container, Label
from qtpy.QtWidgets import QSizePolicy

from skop import OpSpec, ParamSpec

from ._widget import Inputs, build_inputs, param_docs


def is_workflow_plumbing(param: ParamSpec) -> bool:
    """Whether a parameter is drawn by a chooser group rather than on its own.

    True for both halves of a stage: the chooser itself, which becomes a combo
    box, and its arguments dict, which becomes the widgets under it.
    """
    return param.choices is not None or param.params_for is not None


class Stage(Container):
    """One chooser and the inputs of whatever it currently names.

    The combo holds labels rather than functions. magicgui can put arbitrary
    objects in a ComboBox, but the label is the whole point of ``Choices`` --
    "gpu" is what the author wanted shown, not ``richardson_lucy_cupy``.
    """

    def __init__(
        self,
        param: ParamSpec,
        args_param: ParamSpec | None,
        annotation_for: Callable[[ParamSpec], Any],
        value_for: Callable[[ParamSpec, Any], Any],
        tooltip: str | None = None,
    ) -> None:
        self._param = param
        self._args_name = args_param.name if args_param else None
        self._binds = (
            frozenset(args_param.params_for.binds) if args_param else frozenset()
        )
        self._value_for = value_for
        self._annotation_for = annotation_for
        self._doc = tooltip
        self._inputs: Inputs | None = None

        choices = param.choices
        # The default is a function; the combo needs the name this list gives
        # it. An op defaulted to something outside its own Choices is legal
        # in a script (the list constrains the GUI, not the function), and
        # here it simply means the first entry is what comes up.
        current = choices.label(param.default) or choices.labels[0]

        self._combo = ComboBox(
            name=param.name,
            label=param.name.replace("_", " "),
            choices=list(choices.labels),
            value=current,
            tooltip=tooltip,
        )
        # A heading above each stage, so a column of twenty spin boxes reads
        # as the three sub-ops it is rather than as one long form. "Which of
        # these parameters belong to the segmenter" has to be answerable at a
        # glance.
        #
        # The line is the widget's own top border rather than an <hr>, because
        # an <hr> is rich text laid out inside the label and stops at the text's
        # width -- 73px of a 309px panel, which reads as a typo rather than a
        # divider. A border is drawn on the widget, so it spans whatever width
        # the widget has, and the size policy below is what makes that the
        # full width.
        self._rule = Label(value=param.name.replace("_", " "))
        self._rule.native.setStyleSheet(
            "QLabel {"
            "  border-top: 1px solid palette(mid);"
            "  margin-top: 8px; padding-top: 6px;"
            "  font-weight: bold;"
            "}"
        )
        self._rule.native.setSizePolicy(
            QSizePolicy.Expanding, self._rule.native.sizePolicy().verticalPolicy()
        )

        self._box = Container(labels=True)
        super().__init__(widgets=[self._rule, self._combo, self._box], labels=False)
        self.name = param.name
        # The heading already names this stage. Left set, the name would also
        # become a label in the column beside it -- the same word twice, and
        # 70px of width taken from the controls. NB: blanking it is not enough
        # on its own, because a Container unifies its label column across every
        # child; the panel keeps stages in a labels=False container of their
        # own, which is what actually reclaims the width.
        self.label = ""

        self._combo.changed.connect(self._rebuild)
        self._rebuild()

    @property
    def op(self) -> Callable:
        """The op the combo currently names."""
        return self._param.choices.op(self._combo.value)

    @property
    def spec(self) -> OpSpec:
        return OpSpec.from_op(self.op)

    @property
    def runnable(self) -> bool:
        return self._inputs is None or self._inputs.runnable

    def notes(self) -> list[str]:
        """What the panel should say about the currently chosen op.

        Same two notes the panel makes about its own inputs, prefixed with the
        stage they belong to -- otherwise "cannot run: no widget for mask"
        gives no clue which of two combo boxes to change.
        """
        if self._inputs is None:
            return []
        where = f"{self._param.name} ({self._combo.value})"
        notes = []
        if self._inputs.defaulted:
            names = ", ".join(name for name, _ in self._inputs.defaulted)
            notes.append(f"{where}: not editable here, using defaults: {names}")
        if self._inputs.blocking:
            names = ", ".join(name for name, _ in self._inputs.blocking)
            notes.append(f"Cannot run: {where} has no widget for {names}")
        return notes

    def contribution(self) -> dict[str, Any]:
        """This stage's slice of the workflow call."""
        args: dict[str, Any] = {self._param.name: self.op}
        if self._args_name is not None:
            args[self._args_name] = self._inputs.values() if self._inputs else {}
        return args

    def _rebuild(self) -> None:
        """Draw the newly chosen op's inputs, minus what the workflow binds."""
        sub = self.spec
        self._inputs = build_inputs(
            sub,
            self._annotation_for,
            self._value_for,
            skip=lambda param: param.name in self._binds,
        )
        self._box.clear()
        self._box.extend(self._inputs.widgets)

        self._combo.tooltip = _stage_tooltip(self._doc, sub)


def _stage_tooltip(doc: str | None, chosen: OpSpec) -> str:
    """What the combo says on hover: the workflow's words, then the choice's.

    Which environment the chosen op runs in belongs here rather than being
    left to be discovered: picking "gpu" for the first time means a build, and
    that is worth knowing before pressing Run rather than four minutes after.
    """
    summary = (chosen.doc or "").strip().splitlines()
    where = "runs on the host" if chosen.is_workflow else f"environment: {chosen.env}"
    lines = [line for line in (doc, summary[0] if summary else None) if line]
    lines.append(f"({chosen.function} -- {where})")
    return "\n".join(lines)


def stages_for(
    spec: OpSpec,
    annotation_for: Callable[[ParamSpec], Any],
    value_for: Callable[[ParamSpec, Any], Any],
) -> list[Stage]:
    """Build one Stage per chooser in *spec*, in declaration order.

    A chooser with no ``ParamsFor`` partner still gets a combo box; it just
    has nothing under it. That is the shape a workflow takes when a stage has
    no settings worth exposing.
    """
    docs = param_docs(spec.doc)
    partners = {
        param.params_for.chooser: param
        for param in spec.inputs
        if param.params_for is not None
    }
    return [
        Stage(
            param,
            partners.get(param.name),
            annotation_for,
            value_for,
            tooltip=docs.get(param.name),
        )
        for param in spec.inputs
        if param.choices is not None
    ]
