"""Working out what a layer's axes actually are.

skop refuses to guess this ([skop design
0005](https://github.com/apposed/scikit-ops/blob/main/docs/design/0005-dimensional-adaptation.md)),
and it is right to: guessing needs a viewer to guess *for*. This module is
that guess, and it is the whole reason napari-skopkit can hand a `(z, y, x)`
stack to a strictly 2-D op.

The rungs run from "someone said so" to "the layer labels its own axes", and
they stop there. An axis nobody has named stays unnamed: napari shows it as
`-3`, `-2`, `-1`, and so do we. Inventing a name would be a claim about the
data that nothing in the data supports, and it is not needed -- an op that
iterates over an unknown axis does the same thing whatever that axis turns out
to be, and skop matches optional slots by name only, so an unnamed axis can
never be silently consumed as something it is not.

What the viewer is displaying is a different kind of thing, and it lives here
too: it says nothing about what an axis is *called*, but it says exactly which
axes the user is looking at, which is what should land in the op's spatial
slots. `displayed` reports that, in both 2-D and 3-D view, and `_plans` turns
it into an initial mapping -- names or no names.

Each answer carries where it came from and whether it was declared or
inferred, because the panel shows both: a user who disagrees types over it,
and what they type is written back onto the layer, so the disagreement is had
once rather than every run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: napari fills unlabelled axes with their own index, as '-3', '-2', '-1'.
#: Those are placeholders, not names, and must not be read as axis labels.
_PLACEHOLDER = re.compile(r"^-?\d+$")

#: Where napari-skopkit stashes a resolved or user-edited answer, so that later
#: runs -- and other plugins -- can read it back.
METADATA_KEY = "skop_axes"


@dataclass(frozen=True)
class AxisGuess:
    """What this layer's axes are, and how much that is worth.

    ``axes`` holds None for any axis that nobody has named, which is what skop
    wants to hear: an unnamed axis matches no slot by name and never draws a
    warning for being fed to one. Use `display` to show it to a human.
    """

    axes: tuple[str | None, ...]
    #: Human-readable provenance, shown in the panel.
    source: str
    #: Whether someone declared this, as opposed to it being inferred from the
    #: shape. The panel shows inferred answers differently; it does not refuse
    #: to use them.
    declared: bool

    def __str__(self) -> str:
        return " ".join(display(self.axes))


def display(axes: tuple[str | None, ...]) -> tuple[str, ...]:
    """Axis labels as a human sees them, unnamed axes as napari names them."""
    ndim = len(axes)
    return tuple(name or str(i - ndim) for i, name in enumerate(axes))


def parse(text: str) -> tuple[str | None, ...]:
    """Read axis labels a human typed, as whitespace- or comma-separated.

    A placeholder typed back unchanged is still a placeholder: someone editing
    `-3 -2 y x` into `t -2 y x` has named one axis, not four.
    """
    parts = [part for part in re.split(r"[,\s]+", text.strip()) if part]
    return tuple(None if _PLACEHOLDER.match(part) else part for part in parts)


def layer_for(viewer: Any, data: Any) -> Any:
    """Find the layer a magicgui layer combo took *data* from.

    napari's ``ImageData`` annotation hands an op the layer's ``.data`` and
    drops the layer itself, which is exactly the object holding the axis
    metadata. The choice stored that same array, so identity finds it again.

    Returns None when there is no match -- no viewer, or a value that came
    from somewhere else -- and resolution simply falls through to the layer's
    own labels, which without a layer is no labels at all.
    """
    if viewer is None or data is None:
        return None
    for layer in getattr(viewer, "layers", ()):
        if layer.data is data:
            return layer
    return None


def resolve(data: Any, layer: Any = None, viewer: Any = None) -> AxisGuess:
    """Work out the axes of *data*, best evidence first.

    Args:
        data: The array itself, which may be an xarray carrying its own dims.
        layer: The napari layer it came from, if known.
        viewer: The viewer, for deciding whether a trailing extent of 3 is a
            channel axis or a spatial one it happens to be displaying.

    Returns:
        Always an answer, though possibly one that names nothing at all. The
        axes it does not name are None rather than invented.
    """
    ndim = len(getattr(data, "shape", ()) or ())
    for rung in (_from_metadata, _from_xarray, _from_ngff):
        found = rung(data, layer)
        if found is not None and len(found.axes) == ndim:
            return found
    return _from_layer(data, layer, viewer, ndim)


def remember(layer: Any, axes: tuple[str | None, ...]) -> None:
    """Write a resolution back onto the layer.

    The point of the exercise: inference gets *better* over a session rather
    than being redone identically every run, and a correction a user made once
    stays made.

    Only a complete answer is worth keeping. Writing back a partly unnamed one
    would have the next run read it off rung one as though somebody had
    declared it.
    """
    if layer is not None and axes and all(axes):
        layer.metadata[METADATA_KEY] = tuple(axes)


def displayed(viewer: Any, ndim: int) -> tuple[int, ...]:
    """Which of this layer's axes the viewer is showing, innermost last.

    Two of them in 2-D view and three in 3-D, and both are read: which axes
    are on screen is a fact about what the user is looking at either way, and
    running on what they can see is the entire job. It says nothing about what
    those axes are *called*, which is why it is no longer part of naming.

    Empty when the viewer is showing axes this layer does not have -- a
    viewer whose dimensionality has not caught up with this layer, where there
    is no plane to read off.
    """
    dims = getattr(viewer, "dims", None)
    shown = getattr(dims, "displayed", None) or ()
    offset = _offset(dims, ndim)
    axes = tuple(int(i) - offset for i in shown)
    return axes if all(0 <= i < ndim for i in axes) else ()


def positions(axes: tuple[str | None, ...], viewer: Any) -> dict[int, int]:
    """Where the viewer's sliders currently sit, per axis index.

    This is what makes "run on the current Z slice" mean the slice the user is
    actually looking at, rather than slice zero. By index rather than by name,
    because an axis nobody has named still has a slider; skop takes either.
    """
    steps = getattr(getattr(viewer, "dims", None), "current_step", ()) or ()
    offset = len(steps) - len(axes)
    if offset < 0:
        return {}
    return {i: int(steps[offset + i]) for i in range(len(axes))}


# -- the rungs -----------------------------------------------------------


def _from_metadata(data: Any, layer: Any) -> AxisGuess | None:
    """Rung 1: somebody already answered this, here or in an earlier run."""
    stored = getattr(layer, "metadata", {}).get(METADATA_KEY) if layer else None
    if not stored:
        return None
    axes = parse(stored) if isinstance(stored, str) else _named(stored)
    if not any(axes):
        # An earlier session's placeholders, or somebody else's blanks. Either
        # way nobody answered anything, so this rung has nothing to say.
        return None
    return AxisGuess(axes, "layer metadata", declared=all(axes))


def _from_xarray(data: Any, layer: Any) -> AxisGuess | None:
    """Rung 2a: the array brought its own dimension names."""
    dims = getattr(data, "dims", None)
    if not isinstance(dims, tuple) or not all(isinstance(d, str) for d in dims):
        return None
    return AxisGuess(dims, "array dims", declared=True)


def _from_ngff(data: Any, layer: Any) -> AxisGuess | None:
    """Rung 2b: OME-NGFF wrote the axes into the layer's metadata."""
    multiscales = getattr(layer, "metadata", {}).get("multiscales") if layer else None
    try:
        entries = multiscales[0]["axes"]
    except (TypeError, LookupError):
        return None
    names = tuple(
        entry["name"] if isinstance(entry, dict) else str(entry) for entry in entries
    )
    return AxisGuess(names, "NGFF metadata", declared=True) if names else None


def _from_layer(data: Any, layer: Any, viewer: Any, ndim: int) -> AxisGuess:
    """Rung 3: the layer's own axis labels, and whatever the shape settles.

    The last rung, and the only one allowed to answer partly. Whatever it
    cannot name stays None: napari will keep calling that axis -3, and so will
    the panel, until somebody says otherwise.
    """
    names = _labelled(layer, ndim)
    channel, decisive = _channel(data, layer, viewer, ndim)
    guessed = False
    if channel is not None and names[channel] is None:
        names[channel] = "c"
        guessed = not decisive

    return AxisGuess(
        tuple(names),
        "layer axis labels" if any(names) else "napari's own numbering",
        declared=all(names) and not guessed,
    )


def _named(values: Any) -> tuple[str | None, ...]:
    """Labels somebody stored, with blanks and placeholders read as unnamed."""
    return tuple(
        None
        if value is None or not str(value).strip() or _PLACEHOLDER.match(str(value))
        else str(value)
        for value in values
    )


def _labelled(layer: Any, ndim: int) -> list[str | None]:
    """What the layer calls its own axes, placeholders read as unnamed."""
    labels = (getattr(layer, "axis_labels", None) or ()) if layer is not None else ()
    return list(_named(labels)) if len(labels) == ndim else [None] * ndim


def _channel(data: Any, layer: Any, viewer: Any, ndim: int) -> tuple[int | None, bool]:
    """The trailing channel axis, if there is one, and whether that is certain.

    ``rgb`` is the one piece of napari layer state that is decisive rather
    than suggestive: napari only sets it for a trailing channel axis. A
    trailing extent of 3 or 4 is merely likely -- likely enough that
    ``skop.ops._util.to_gray`` already assumes it -- so it is offered as an
    inference. And if the viewer is displaying that axis, it is spatial, and
    calling it a channel would transpose the image out from under the user.

    This is the only name inferred from the data rather than declared, and c
    is the one worth inferring: it is the axis ops routinely declare as
    optional, and skop fills an optional slot by name and never by position
    ([skop design 0006]). Unnamed, an RGB axis gets iterated over three times
    instead of being handed to `to_gray` whole.
    """
    if getattr(layer, "rgb", False) and ndim:
        return ndim - 1, True
    shape = tuple(getattr(data, "shape", ()) or ())
    if ndim >= 3 and shape[-1] in (3, 4) and ndim - 1 not in displayed(viewer, ndim):
        return ndim - 1, False
    return None, False


def _offset(dims: Any, ndim: int) -> int:
    """How many of the viewer's outer axes this layer does not have.

    napari's dims belong to the viewer and are shared across its layers,
    right-aligned onto each: a 3-D layer in a 4-D viewer has the viewer's axis
    1 as its own axis 0.
    """
    viewer_ndim = getattr(dims, "ndim", None)
    if not viewer_ndim:
        viewer_ndim = len(getattr(dims, "current_step", ()) or ()) or ndim
    return int(viewer_ndim) - ndim
