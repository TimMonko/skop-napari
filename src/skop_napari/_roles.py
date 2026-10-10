"""Translating skop's roles into napari's vocabulary.

This is the whole of what makes skop-napari napari-specific: two lookup
tables and one shape adapter. Everything else in this package is about
widgets and threads.

skop deliberately refuses to guess -- an unannotated array reports no role at
all -- so the guessing happens here, where there is a viewer to guess for.

The adapters exist because a role names what an array *means*, not how napari
wants it laid out. skop states bounding boxes as ``(N, 4)`` rows of
``[min_y, min_x, max_y, max_x]``; napari reads that same array as a single
shape with N vertices in four dimensions. Reconciling the two is this
package's job, not skop's -- skop.types imports no GUI, and the layout napari
wants is not more correct, only more napari.

Both directions need it, which is easy to forget: ``layer_args_for`` adapts an
op's output on the way to a layer, and ``value_for`` adapts a layer's data on
the way into an op. A mask detector taking boxes from a Shapes layer is the
case that needs both.
"""

from __future__ import annotations

from typing import Any

import napari.types as nt
import numpy as np
from ophelper import boxes, masks

from skop import OutputSpec, ParamSpec, Role

#: How a stack of masks can be shown, since no layer holds one directly.
#: Ordered largest object first in every case, so the label a given object
#: gets does not depend on which projection is chosen.
#:
#: The two 2-D entries differ only in who wins a pixel two masks claim. With
#: largest-first order the biggest object holds label 1, so ``min`` hands
#: contested pixels to it -- and an object drawn wholly inside another then
#: disappears from the picture. ``max`` draws the nested one on top instead,
#: which is why it leads: a projection that silently deletes objects is a
#: poor thing to show someone first.
MASK_VIEWS: dict[str, str] = {
    "2D labels (nested objects on top)": "max",
    "2D labels (largest object on top)": "min",
    "3D stack (nothing lost)": "3d",
}
DEFAULT_MASK_VIEW = next(iter(MASK_VIEWS))

#: How far apart to space the planes of a 3-D mask stack, relative to a pixel.
#: Ten rather than one because the axis is an object index, not a depth: ten
#: masks over a 512-pixel image at unit spacing is a pancake, and rotating it
#: shows nothing. This only affects how the stack is drawn.
DEFAULT_Z_SPACING = 10.0


def is_stack_view(mask_view: str) -> bool:
    """Whether *mask_view* asks for the 3-D stack rather than a projection."""
    return MASK_VIEWS.get(mask_view, "min") == "3d"


# What magicgui should build an input widget for. The napari.types aliases
# give a layer combo box that hands the op the layer's .data, which is the
# ndarray the op wanted in the first place.
_INPUT_TYPES: dict[Role, Any] = {
    Role.image: nt.ImageData,
    Role.labels: nt.LabelsData,
    Role.points: nt.PointsData,
    # Boxes are drawn on a Shapes layer, and read back off one.
    Role.boxes: nt.ShapesData,
    Role.shapes: nt.ShapesData,
    Role.surface: nt.SurfaceData,
    Role.tracks: nt.TracksData,
    Role.vectors: nt.VectorsData,
}

# What kind of layer an output becomes, as Layer.create() names them.
#
# Role.masks is the one that is not a straight lookup. napari has no Masks
# layer, so a stack of masks becomes a Labels layer by way of a projection --
# which one is the user's choice, and layer_args_for applies it. Both the 2-D
# and 3-D answers are Labels layers, so the entry itself is unconditional.
_LAYER_TYPES: dict[Role, str] = {
    Role.image: "image",
    Role.labels: "labels",
    Role.masks: "labels",
    Role.points: "points",
    Role.boxes: "shapes",
    Role.shapes: "shapes",
    Role.surface: "surface",
    Role.tracks: "tracks",
    Role.vectors: "vectors",
}


def annotation_for(param: ParamSpec) -> Any:
    """The type magicgui should build a widget for, given an op parameter."""
    if param.role is not None:
        return _INPUT_TYPES[param.role]
    if param.type is np.ndarray:
        # An unannotated array is most likely a picture, and offering the
        # wrong layer combo beats offering no way to pass one at all.
        return nt.ImageData
    return param.type


def layer_type_for(output: OutputSpec) -> str | None:
    """The napari layer type for an op output, or None if it is not a layer.

    Scalars -- unseg's nucleus and cell counts, say -- have no layer to be,
    and belong in a results panel instead.
    """
    if output.role is not None:
        return _LAYER_TYPES[output.role]
    if output.type is np.ndarray:
        return "image"
    return None


def value_for(param: ParamSpec, value: Any) -> Any:
    """What an op wants, given what the widget for *param* is holding.

    The mirror of :func:`layer_args_for`, and needed for the same reason: a
    Shapes layer's ``.data`` is a list of vertex arrays, one per shape, and an
    op that asked for ``(N, 4)`` boxes gets ``(11, 4, 2)`` without this.
    ``ophelper.boxes.from_napari`` takes both that list and the array napari
    sometimes makes of it.

    An array already in skop's layout passes through, so an op called with
    boxes from somewhere other than a layer is not converted twice.
    """
    if param.role is Role.boxes and value is not None:
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[-1] == 4:
            return value
        return boxes.from_napari(value)
    return value


def layer_args_for(
    output: OutputSpec,
    value: Any,
    mask_view: str = DEFAULT_MASK_VIEW,
    z_spacing: float = DEFAULT_Z_SPACING,
) -> tuple[Any, dict[str, Any]]:
    """The data and extra layer keywords for an op output.

    Returns the value unchanged for every role whose layout napari already
    agrees with, which is all of them but two.

    **Bounding boxes.** skop states them as ``(N, 4)`` rows of
    ``[min_y, min_x, max_y, max_x]`` -- see ``ophelper.boxes`` -- and a Shapes
    layer handed that array reads it as one N-vertex shape in 4-D and raises.
    ``ophelper.boxes.to_napari`` reshapes it into the ``(N, 2, 2)`` corner pairs a
    rectangle wants, but only with ``shape_type`` said out loud: napari's
    default for a 2-vertex shape is a rectangle that then complains it was
    given two corners rather than four.

    **Masks.** An ``(N, Y, X)`` stack of possibly-overlapping masks is not a
    layer at all, so *mask_view* picks the projection to show it through: a
    2-D label image, which is lossy where masks overlap, or the 3-D stack,
    which loses nothing and can be rotated. Masks are sorted largest-first
    before projecting, which is what gives the 2-D strategies their meaning --
    see ``MASK_VIEWS``.
    """
    if output.role is Role.boxes:
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[-1] == 4:
            return boxes.to_napari(array), {
                "shape_type": "rectangle",
                # A Shapes layer defaults to solid white faces, which would
                # cover the very thing the boxes are pointing at.
                "face_color": "transparent",
                "edge_color": "yellow",
                "edge_width": 2,
            }

    if output.role is Role.masks:
        stack = np.asarray(value)
        ordered = stack[masks.order_by_area(stack)] if len(stack) else stack
        if is_stack_view(mask_view):
            # The stack's first axis is an object index, not a depth, so its
            # spacing is a viewing choice with no true value -- see
            # DEFAULT_Z_SPACING. y and x stay at 1, matching the image the
            # masks were found in.
            #
            # NB: set once, when the layer is created. Changing the spinner
            # after the fact re-runs the op rather than restyling the layer,
            # which is the next piece of work here.
            return masks.to_labels_3d(ordered), {"scale": (z_spacing, 1.0, 1.0)}
        return masks.to_labels_2d(ordered, MASK_VIEWS.get(mask_view, "min")), {}

    return value, {}
