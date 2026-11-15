"""Turning an OpSpec into magicgui input widgets.

Nothing here imports napari. The two napari-shaped things a caller must supply
are ``annotation_for``, which says what type each parameter should be rendered
as, and ``value_for``, which converts what the resulting widget holds into
what the op asked for. ``_roles`` has the napari answers, and a different
front end would pass different ones. If skop ever grows a second magicgui
host, this module is what moves.

Widgets are built one parameter at a time rather than by handing magicgui a
whole synthesized function, because ops in the wild contain parameters
magicgui cannot render -- ``tuple[int, ...]`` among them -- and one of those
must not take the entire panel down with it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from magicgui.types import Undefined
from magicgui.widgets import Widget, create_widget
from opspec.op import OpSpec, ParamSpec


def _unconverted(param: ParamSpec, value: Any) -> Any:
    """The default conversion: none at all."""
    return value


@dataclass
class Inputs:
    """The renderable inputs of an op, and what had to be left out."""

    widgets: list[Widget] = field(default_factory=list)
    #: Parameters magicgui could not render, which have a default to fall
    #: back on. The op still runs; these simply are not adjustable.
    defaulted: list[tuple[str, str]] = field(default_factory=list)
    #: Parameters magicgui could not render and which have no default. The
    #: op cannot be run from a GUI at all.
    blocking: list[tuple[str, str]] = field(default_factory=list)
    #: The spec behind each widget, kept so that values() can ask what a
    #: parameter meant rather than guessing from what the widget holds.
    params: dict[str, ParamSpec] = field(default_factory=dict)
    #: How to turn a widget's value into what the op wants. Supplied by the
    #: front end, because the layout a widget holds is the front end's.
    convert: Callable[[ParamSpec, Any], Any] = _unconverted
    #: Value sources that are not a single widget holding a single parameter
    #: -- the chooser groups a workflow gets. Each supplies a slice of the
    #: call through ``contribution()`` and reports its own ``runnable``.
    #: Duck-typed rather than imported, because the module that builds them
    #: builds on this one.
    extra: list[Any] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        # A chosen sub-op with an unrenderable required input stops the
        # workflow just as surely as one of the workflow's own would.
        return not self.blocking and all(group.runnable for group in self.extra)

    def values(self) -> dict[str, Any]:
        """The arguments to call the op with, as currently filled in."""
        # A widget with no ParamSpec is one of the composite widgets above,
        # laid out here but reporting its value through `extra`.
        args = {
            w.name: self.convert(self.params[w.name], w.value)
            for w in self.widgets
            if w.name in self.params
        }
        for group in self.extra:
            args.update(group.contribution())
        return args


def build_inputs(
    spec: OpSpec,
    annotation_for: Callable[[ParamSpec], Any],
    value_for: Callable[[ParamSpec, Any], Any] = _unconverted,
    skip: Callable[[ParamSpec], bool] | None = None,
) -> Inputs:
    """Build one widget per input parameter of *spec*.

    ``Out`` parameters are skipped: they are buffers the caller allocates, and
    a user is never asked for one. A caller may skip more through *skip* --
    which is how a workflow leaves out the parameters it fills in itself, and
    how the sub-op parameters it binds are kept off the panel.
    """
    docs = param_docs(spec.doc)
    inputs = Inputs(convert=value_for)

    for param in spec.inputs:
        if skip is not None and skip(param):
            continue
        options = dict(param.ui)
        widget_type = options.pop("widget_type", None)
        if "tooltip" not in options and param.name in docs:
            options["tooltip"] = docs[param.name]

        try:
            widget = create_widget(
                # NB: Undefined, not None. Passing None marks the widget
                # nullable, which puts an empty entry in every layer combo.
                value=Undefined if param.required else param.default,
                annotation=annotation_for(param),
                name=param.name,
                label=param.name.replace("_", " "),
                widget_type=widget_type,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001 - any failure is the same failure.
            reason = f"{type(exc).__name__}: {exc}"
            if param.required:
                inputs.blocking.append((param.name, reason))
            else:
                inputs.defaulted.append((param.name, reason))
            continue

        inputs.widgets.append(widget)
        inputs.params[param.name] = param

    return inputs


_ARGS_HEADING = re.compile(r"^\s*(Args|Arguments|Parameters)\s*:\s*$")
_SECTION_HEADING = re.compile(r"^\s*[A-Z][A-Za-z ]*\s*:\s*$")
_ARG_LINE = re.compile(r"^\s*(\*{0,2}\w+)\s*(\([^)]*\))?\s*:\s*(.*)$")


def param_docs(doc: str | None) -> dict[str, str]:
    """Pull per-parameter descriptions out of a Google-style docstring.

    magicgui does this itself when it builds from a function, but we build
    from an OpSpec, so the tooltips have to be recovered here. A docstring
    that does not follow the convention simply yields no tooltips.
    """
    if not doc:
        return {}

    lines = doc.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if _ARGS_HEADING.match(line))
    except StopIteration:
        return {}

    found: dict[str, str] = {}
    current: str | None = None
    for line in lines[start + 1 :]:
        if not line.strip():
            continue
        if _SECTION_HEADING.match(line):
            break
        match = _ARG_LINE.match(line)
        if match:
            current = match.group(1)
            found[current] = match.group(3).strip()
        elif current:
            # A continuation of the previous parameter's description.
            found[current] = f"{found[current]} {line.strip()}".strip()

    return found
