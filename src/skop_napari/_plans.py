"""Choosing how an array gets fitted to the op that is about to receive it.

An op says how many axes it consumes and what it likes to call them; skop turns
that plus "this array is `z y x`" into one `AdaptationPlan`, best effort, and
lets it be edited ([skop design
0006](https://github.com/apposed/scikit-ops/blob/main/docs/design/0006-axis-mapping.md)).
This is the editing: one row per adaptable image input, with the resolved axes
on top, a mapping combo per slot the op consumes, and a disposition combo per
axis left over.

The rule the layout encodes is that skop no longer forbids anything. The
default plan never discards data, a misaligned mapping is shown as a warning
rather than a blocked run, and the only thing that disables Run is an array
that cannot be adapted at all.
"""

from __future__ import annotations

from typing import Any

from magicgui.widgets import ComboBox, Container, Label, LineEdit
from opspec.op import OpSpec, ParamSpec
from opspec.plan import ITERATE, PASS, SELECT, AdaptationPlan, plan
from psygnal import Signal

from . import _axes

#: Shown in a mapping combo for an optional slot nobody is filling.
_NONE = "—"

#: Slots that are about where something is in space, which is exactly what the
#: viewer's displayed axes are about. Everything else -- c, t -- is about what
#: the data means, which layout says nothing about.
_SPATIAL = ("z", "y", "x")


class _Row(Container):
    """Axis labels, slot mapping and leftover handling for one image input."""

    #: Emitted when anything in this row is changed by hand. Deliberately not
    #: Container.changed, which also fires while the row is being rebuilt --
    #: rendering a plan must not trigger the re-planning that rendered it.
    edited = Signal()

    def __init__(self, param: ParamSpec) -> None:
        self.param = param
        self._override: tuple[str | None, ...] | None = None
        self._mapping: tuple[int | None, ...] | None = None
        self._dispositions: dict[int, str] | None = None
        self._seen: int | None = None
        self._updating = False
        self._plan: AdaptationPlan | None = None
        self._problem: str | None = None

        self._axes = LineEdit(
            name=f"{param.name}_axes",
            label=f"{param.name} axes",
            tooltip=(
                "What this array's axes are, outermost first. Any name works: "
                "'z y x', 'lifetime y x', 'pln row col'. Edit this if the "
                "guess is wrong -- it is remembered on the layer."
            ),
        )
        self._slots = Container(layout="vertical", labels=True)
        self._extra = Container(layout="vertical", labels=True)
        self._status = Label(value="")
        self._warn = Label(value="")

        super().__init__(
            widgets=[self._axes, self._slots, self._extra, self._status, self._warn],
            labels=True,
            visible=False,
        )
        self._axes.changed.connect(self._axes_edited)

    # -- state ------------------------------------------------------------

    @property
    def plan(self) -> AdaptationPlan | None:
        return self._plan

    @property
    def problem(self) -> str | None:
        """Why this input cannot be adapted, if it cannot."""
        return self._problem

    @property
    def warnings(self) -> tuple[str, ...]:
        """Slots being fed something other than what they asked for."""
        return self._plan.warnings if self._plan else ()

    def _axes_edited(self) -> None:
        if self._updating:
            return
        # Typed by hand, so it outranks anything resolution comes up with --
        # until the input layer changes underneath it. The axes changing also
        # invalidates any mapping, whose indices referred to the old ones.
        self._override = _axes.parse(self._axes.value) or None
        self._forget_choices()
        self.edited.emit()

    def _forget_choices(self) -> None:
        self._mapping = None
        self._dispositions = None

    # -- refreshing -------------------------------------------------------

    def refresh(self, fn: Any, data: Any, viewer: Any) -> None:
        """Re-resolve axes and re-plan, for whatever is selected now."""
        if data is None or not hasattr(data, "shape"):
            self.visible = False
            self._problem = None
            self._plan = None
            return

        if id(data) != self._seen:
            # A different layer: choices made for the old one say nothing
            # about this one, and its axis indices may not even line up.
            self._seen = id(data)
            self._override = None
            self._forget_choices()

        layer = _axes.layer_for(viewer, data)
        if self._override is not None:
            axes, note = self._override, "as you typed"
        else:
            guess = _axes.resolve(data, layer, viewer)
            axes, note = guess.axes, f"from {guess.source}"
            if not guess.declared:
                note += " -- check this"

        self._show_axes(axes)
        _axes.remember(layer, axes)
        self._replan(fn, data, axes, viewer, note)
        self.visible = True

    def _show_axes(self, axes: tuple[str | None, ...]) -> None:
        self._updating = True
        try:
            self._axes.value = " ".join(_axes.display(axes))
        finally:
            self._updating = False

    def _replan(
        self,
        fn: Any,
        data: Any,
        axes: tuple[str | None, ...],
        viewer: Any,
        note: str,
    ) -> None:
        try:
            plan = self._planned(fn, data, axes, viewer, self._mapping)
            if self._mapping is None:
                # Nobody has chosen by hand, so the viewer gets to: point the
                # spatial slots at whatever is on screen. Re-planned rather
                # than patched, because the mapping decides what is left over
                # and hence what the dispositions apply to.
                # napari does not count an RGB layer's colour axis as a
                # dimension, so it is never on screen; leave it out, or the
                # viewer's two axes land on the last two, x and colour.
                layer = _axes.layer_for(viewer, data)
                rgb = bool(getattr(layer, "rgb", False))
                guided = _guided(
                    plan.mapping,
                    self.param.axes.slots,
                    _axes.displayed(viewer, len(axes) - rgb),
                    len(axes),
                )
                if guided != plan.mapping:
                    plan = self._planned(fn, data, axes, viewer, guided)
        except ValueError as exc:
            # An array that cannot satisfy the op at all: the wrong number of
            # labels, or fewer axes than the op consumes. The user's move is
            # to fix the labels, so say so here rather than at run time.
            self._problem = f"{self.param.name}: {exc}"
            self._plan = None
            self._slots.clear()
            self._extra.clear()
            self._status.value = str(exc)
            self._warn.value = ""
            return

        self._problem = None
        self._plan = plan
        self._status.value = f"{note} -- {plan.summary}"
        self._warn.value = "; ".join(plan.warnings)
        self._render(plan, axes, tuple(data.shape))

    def _planned(
        self,
        fn: Any,
        data: Any,
        axes: tuple[str | None, ...],
        viewer: Any,
        mapping: tuple[int | None, ...] | None,
    ) -> AdaptationPlan:
        return plan(
            fn,
            self.param.name,
            data,
            axes,
            position=_axes.positions(axes, viewer),
            mapping=mapping,
            dispositions=self._dispositions,
        )

    # -- the controls -----------------------------------------------------

    def _render(
        self, plan: AdaptationPlan, axes: tuple[str | None, ...], shape: tuple[int, ...]
    ) -> None:
        """Rebuild the mapping and disposition combos to match *plan*.

        Rebuilt wholesale rather than patched, because unmapping a slot moves
        an axis from one group to the other: which controls exist at all is a
        function of the plan.
        """
        self._updating = True
        try:
            self._slots.clear()
            self._extra.clear()
            # Unnamed axes are offered under the names napari gives them, so
            # that a combo entry and a slider label say the same thing.
            shown = _axes.display(axes)
            choices = [(shown[i], i) for i in range(len(axes))]

            for index, slot in enumerate(self.param.axes.slots):
                combo = ComboBox(
                    name=f"{self.param.name}_slot{index}",
                    label=f"{slot} ←",
                    choices=choices + ([(_NONE, None)] if slot.optional else []),
                    value=plan.mapping[index],
                    tooltip=f"Which of this array's axes to feed the op's {slot} axis.",
                )
                combo.changed.connect(
                    lambda value, s=index: self._slot_chosen(s, value)
                )
                self._slots.append(combo)

            selected = dict(plan.select)
            for i in range(len(axes)):
                if i in plan.mapping:
                    continue
                options = [
                    (f"iterate all {shape[i]}", ITERATE),
                    ("current position", SELECT),
                ]
                if self.param.axes.variadic:
                    options.append(("hand to the op", PASS))
                current = (
                    ITERATE if i in plan.iterate else SELECT if i in selected else PASS
                )
                combo = ComboBox(
                    name=f"{self.param.name}_extra{i}",
                    label=shown[i],
                    choices=options,
                    value=current,
                    tooltip=f"What to do with the {shown[i]} axis, "
                    "which the op does not consume.",
                )
                combo.changed.connect(
                    lambda value, axis=i: self._disposition_chosen(axis, value)
                )
                self._extra.append(combo)

            self._slots.visible = bool(len(self._slots))
            self._extra.visible = bool(len(self._extra))
        finally:
            self._updating = False

    def _slot_chosen(self, slot: int, value: int | None) -> None:
        if self._updating or self._plan is None:
            return
        self._mapping = _reassign(
            self._plan.mapping,
            slot,
            value,
            self.param.axes.slots,
            len(self._plan.input_axes),
        )
        self.edited.emit()

    def _disposition_chosen(self, axis: int, value: str) -> None:
        if self._updating or self._plan is None:
            return
        chosen = dict(self._dispositions or {})
        chosen[axis] = value
        self._dispositions = chosen
        self.edited.emit()


def _guided(
    mapping: tuple[int | None, ...],
    slots: tuple[Any, ...],
    displayed: tuple[int, ...],
    naxes: int,
) -> tuple[int | None, ...]:
    """Point the op's spatial slots at the axes the viewer is displaying.

    Which axes are on screen is a fact about what the user is looking at, and
    it outranks a name: somebody who has rolled the dims round to look at the
    zx plane means to run on the zx plane, whatever the axes are called. skop
    says so in a warning rather than refusing, which is the right volume for
    it -- and in the ordinary view, where nobody has rolled anything, layout
    and names agree and this changes nothing.

    Both lists are right-aligned, innermost last, so a 2-D op in a 3-D view
    takes the two innermost displayed axes, and a 3-D op in a 2-D view keeps
    its z from wherever the names put it.
    """
    spatial = [s for s, slot in enumerate(slots) if slot.name in _SPATIAL]
    count = min(len(spatial), len(displayed))
    if not count:
        return mapping

    forced = dict(zip(spatial[-count:], displayed[-count:]))
    new = list(mapping)
    for slot, axis in forced.items():
        new[slot] = axis

    used = set(forced.values())
    for slot in range(len(new)):
        if slot in forced or new[slot] is None:
            continue
        if new[slot] not in used:
            used.add(new[slot])
            continue
        # Something else was holding an axis that is now on screen. An
        # optional slot goes empty rather than being handed an axis it never
        # asked for -- filling c by position is exactly what skop design 0006
        # forbids -- and a required one takes whatever is still going.
        free = next((i for i in range(naxes) if i not in used), None)
        new[slot] = None if slots[slot].optional else free
        if new[slot] is not None:
            used.add(new[slot])

    if any(index is None and not slot.optional for slot, index in zip(slots, new)):
        return mapping  # Nothing gained by handing skop a mapping it refuses.
    return tuple(new)


def _reassign(
    mapping: tuple[int | None, ...],
    slot: int,
    value: int | None,
    slots: tuple[Any, ...],
    naxes: int,
) -> tuple[int | None, ...]:
    """Put *value* in *slot*, swapping out whichever slot already held it.

    A mapping is a permutation, so choosing an axis that is already spoken for
    has to displace something. Swapping is what a user means by it: dragging z
    into the y slot puts whatever was in y where z came from.
    """
    new = list(mapping)
    old = new[slot]
    new[slot] = value
    if value is None:
        return tuple(new)
    for other, held in enumerate(new):
        if other == slot or held != value:
            continue
        new[other] = old
        if old is None and not slots[other].optional:
            # Nothing to swap back, and that slot must be filled: give it the
            # first axis nobody else is using.
            used = {v for v in new if v is not None}
            free = [j for j in range(naxes) if j not in used]
            if free:
                new[other] = free[0]
        break
    return tuple(new)


class Adaptations(Container):
    """The adaptation rows for whichever op is currently selected."""

    def __init__(self) -> None:
        super().__init__(labels=False, visible=False)
        self._rows: list[_Row] = []

    def rebuild(self, spec: OpSpec) -> list[str]:
        """Start over for a newly chosen op.

        Returns the names of the parameters that got a row, so the panel can
        watch those widgets for changes.
        """
        self.clear()
        # Only a parameter whose op declared Axes can be adapted; everything
        # else is passed through exactly as it always was.
        self._rows = [_Row(p) for p in spec.inputs if p.axes is not None]
        self.extend(self._rows)
        self.visible = bool(self._rows)
        return [row.param.name for row in self._rows]

    def refresh(self, fn: Any, values: dict[str, Any], viewer: Any) -> None:
        for row in self._rows:
            row.refresh(fn, values.get(row.param.name), viewer)
        self.visible = any(row.visible for row in self._rows)

    @property
    def plans(self) -> dict[str, AdaptationPlan]:
        """The current plan per parameter, for ``Runner.run(plans=...)``."""
        return {row.param.name: row.plan for row in self._rows if row.plan is not None}

    @property
    def problems(self) -> list[str]:
        return [row.problem for row in self._rows if row.problem]

    @property
    def warnings(self) -> list[str]:
        """Misalignments worth showing, none of which block a run."""
        return [
            f"{row.param.name}: {note}" for row in self._rows for note in row.warnings
        ]

    @property
    def output_axes(self) -> tuple[str, ...]:
        """Axes to stamp onto the results, from whichever plan produced them.

        Derived rather than declared, and only trustworthy when exactly one
        input was adapted -- which is every op in the collection today, and
        the same limit skop's own iteration imposes.
        """
        chosen = list(self.plans.values())
        return chosen[0].output_axes if len(chosen) == 1 else ()

    def watch(self, callback: Any) -> None:
        for row in self._rows:
            row.edited.connect(callback)
