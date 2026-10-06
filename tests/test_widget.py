"""Building input widgets from op specs."""

from __future__ import annotations

import napari.types as nt
import numpy as np

import skop
from skop_napari._roles import annotation_for, layer_type_for
from skop_napari._widget import build_inputs, param_docs


def test_roles_become_napari_types():
    from skop.ops.threshold import otsu

    spec = skop.OpSpec.from_op(otsu)
    image = next(p for p in spec.params if p.name == "image")
    assert annotation_for(image) is nt.ImageData
    assert [layer_type_for(o) for o in spec.outputs] == ["labels"]


def test_unannotated_array_is_guessed_to_be_an_image():
    # skop reports role None; guessing is this layer's job, not skop's.
    from skop.ops import toy

    spec = skop.OpSpec.from_op(toy.scale)
    image = next(p for p in spec.params if p.name == "image")
    assert image.role is None
    assert annotation_for(image) is nt.ImageData


def test_scalar_outputs_are_not_layers():
    from skop.ops.segment import unseg

    spec = skop.OpSpec.from_op(unseg)
    kinds = {o.name: layer_type_for(o) for o in spec.outputs}
    assert kinds == {
        "nuclei": "labels",
        "cells": "labels",
        "n_nuclei": None,
        "n_cells": None,
    }


def test_builds_a_widget_per_input(qtbot):
    from skop.ops.segment import stardist2d_fluo

    spec = skop.OpSpec.from_op(stardist2d_fluo)
    inputs = build_inputs(spec, annotation_for)

    assert inputs.runnable
    assert [w.name for w in inputs.widgets] == [
        "image",
        "prob_thresh",
        "nms_thresh",
        "normalize",
    ]
    # The UI hints carried on the op's annotations reach the widget.
    prob = next(w for w in inputs.widgets if w.name == "prob_thresh")
    assert type(prob).__name__ == "FloatSlider"
    assert prob.max == 1.0
    assert prob.value == 0.5


def test_output_buffers_are_never_asked_for(qtbot):
    from skop.ops import toy

    spec = skop.OpSpec.from_op(toy.scale_into)
    inputs = build_inputs(spec, annotation_for)
    # 'result' is an Out param: the caller allocates it, so no widget.
    assert [w.name for w in inputs.widgets] == ["image", "factor"]


def test_unrenderable_optional_falls_back_to_its_default(qtbot):
    # unseg takes tuple[int, ...] parameters, which magicgui cannot render.
    # They have defaults, so the op stays runnable without them.
    from skop.ops.segment import unseg

    spec = skop.OpSpec.from_op(unseg)
    inputs = build_inputs(spec, annotation_for)

    assert inputs.runnable
    assert "nk" in dict(inputs.defaulted)
    assert "nk" not in [w.name for w in inputs.widgets]
    assert "image" in [w.name for w in inputs.widgets]


class Opaque:
    """A type no widget exists for. Module scope, so annotations resolve."""


def test_unrenderable_required_input_blocks_the_op(qtbot):
    @skop.op(env="minimal")
    def awkward(thing: Opaque) -> int:
        return 0

    inputs = build_inputs(skop.OpSpec.from_op(awkward), annotation_for)
    assert not inputs.runnable
    assert [name for name, _ in inputs.blocking] == ["thing"]


def test_docstring_becomes_tooltips(qtbot):
    from skop.ops.threshold import otsu

    docs = param_docs(skop.OpSpec.from_op(otsu).doc)
    assert "trailing RGB(A) axis" in docs["image"]
    # Wrapped continuation lines are joined onto their parameter.
    assert docs["label_objects"].startswith("Whether to label connected components")
    assert "binary mask" in docs["label_objects"]

    inputs = build_inputs(skop.OpSpec.from_op(otsu), annotation_for)
    invert = next(w for w in inputs.widgets if w.name == "invert")
    assert "below the threshold" in invert.tooltip


def test_param_docs_tolerates_an_unconventional_docstring():
    assert param_docs(None) == {}
    assert param_docs("Just a summary, no sections.") == {}


def test_values_reflect_the_widgets(qtbot):
    from skop.ops.threshold import otsu

    inputs = build_inputs(skop.OpSpec.from_op(otsu), annotation_for)
    next(w for w in inputs.widgets if w.name == "invert").value = True

    values = inputs.values()
    assert values["invert"] is True
    assert values["label_objects"] is True
    assert values["image"] is None  # No layer chosen yet.


def test_outputs_are_labeled_by_name():
    from skop.ops import toy
    from skop_napari._run import outputs_of

    single = skop.OpSpec.from_op(toy.add)
    assert outputs_of(single, 42) == {"result": 42}

    multi = skop.OpSpec.from_op(toy.scale)
    scaled = np.zeros((2, 2))
    assert outputs_of(multi, (scaled, 7.0)) == {"scaled": scaled, "total": 7.0}


def test_a_single_field_namedtuple_output_is_unwrapped():
    # An op may return a one-field NamedTuple to name its single output --
    # the box detectors do, so their layer is called "boxes" and not
    # "result". Counting outputs cannot tell that from a bare return, and
    # labeling the tuple itself hands napari a tuple where an array belongs.
    from skop.ops.detect import fastsam
    from skop.ops.detect._result import Boxes
    from skop_napari._run import outputs_of

    boxes = np.array([[236.4, 144.1, 287.5, 201.0]], dtype=np.float32)
    labeled = outputs_of(skop.OpSpec.from_op(fastsam), Boxes(boxes))

    assert list(labeled) == ["boxes"]
    assert labeled["boxes"] is boxes


def test_resolve_finds_the_op_function():
    from skop.ops import toy
    from skop_napari._run import resolve

    assert resolve(skop.OpSpec.from_op(toy.add)) is toy.add


def test_shapes_inputs_are_converted_from_what_a_layer_holds(qtbot):
    """The bug this exists for: a Shapes layer's data is not (N, 4).

    napari hands over one (V, 2) vertex array per shape. A mask detector that
    asked for (N, 4) boxes was getting (11, 4, 2) and refusing it in the
    worker, several seconds and one model load later.
    """
    from skop import Role
    from skop.ops.mask import mobilesam_masks
    from skop_napari._roles import value_for

    boxes_param = next(
        p for p in skop.OpSpec.from_op(mobilesam_masks).params if p.name == "boxes"
    )
    assert boxes_param.role is Role.boxes

    # Exactly what napari gives: a list of rectangles, four corners each.
    layer_data = [
        np.array([[20, 10], [20, 30], [40, 30], [40, 10]], dtype=np.float32),
        np.array([[0, 0], [0, 5], [5, 5], [5, 0]], dtype=np.float32),
    ]
    converted = value_for(boxes_param, layer_data)
    assert converted.shape == (2, 4)
    assert converted.tolist() == [[20, 10, 40, 30], [0, 0, 5, 5]]


def test_canonical_boxes_pass_through_unconverted(qtbot):
    # An op called with boxes from somewhere other than a layer must not be
    # converted twice.
    from skop.ops.mask import mobilesam_masks
    from skop_napari._roles import value_for

    boxes_param = next(
        p for p in skop.OpSpec.from_op(mobilesam_masks).params if p.name == "boxes"
    )
    canonical = np.array([[20, 10, 40, 30]], dtype=np.float32)
    assert value_for(boxes_param, canonical) is canonical
