"""Working out what a layer's axes are, in order of how much the answer is worth."""

from __future__ import annotations

import numpy as np
import pytest

from napari_skopkit import _axes


class FakeDims:
    def __init__(self, displayed=(), current_step=(), ndim=None):
        self.displayed = displayed
        self.current_step = current_step
        self.ndim = ndim


class FakeViewer:
    def __init__(self, displayed=(), current_step=(), layers=(), ndim=None):
        self.dims = FakeDims(displayed, current_step, ndim)
        self.layers = list(layers)


class FakeLayer:
    """Just the layer attributes resolution reads."""

    def __init__(self, data, *, metadata=None, axis_labels=None, rgb=False):
        self.data = data
        self.metadata = dict(metadata or {})
        self.axis_labels = axis_labels
        self.rgb = rgb


# -- rung 1: somebody already answered ------------------------------------


def test_metadata_outranks_everything():
    data = np.zeros((5, 4, 6))
    layer = FakeLayer(data, metadata={_axes.METADATA_KEY: ("lifetime", "y", "x")})
    guess = _axes.resolve(data, layer)
    assert guess.axes == ("lifetime", "y", "x")
    assert guess.declared


def test_metadata_may_have_been_written_as_a_string():
    data = np.zeros((3, 4))
    layer = FakeLayer(data, metadata={_axes.METADATA_KEY: "y x"})
    assert _axes.resolve(data, layer).axes == ("y", "x")


def test_a_resolution_is_written_back_to_the_layer():
    # The point of the exercise: guess once per layer, not once per run.
    layer = FakeLayer(np.zeros((4, 4)))
    _axes.remember(layer, ("y", "x"))
    assert layer.metadata[_axes.METADATA_KEY] == ("y", "x")


# -- rung 2: the data or its provenance says -------------------------------


def test_xarray_dims_are_taken_at_their_word():
    xr = pytest.importorskip("xarray")
    data = xr.DataArray(np.zeros((5, 4, 6)), dims=("pln", "row", "col"))
    guess = _axes.resolve(data)
    assert guess.axes == ("pln", "row", "col")
    assert guess.declared
    # skop resolves the synonyms; this layer does not second-guess them.


def test_ngff_axes_are_read_from_layer_metadata():
    data = np.zeros((2, 5, 4, 6))
    layer = FakeLayer(
        data,
        metadata={
            "multiscales": [
                {
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space"},
                        {"name": "x", "type": "space"},
                    ]
                }
            ]
        },
    )
    assert _axes.resolve(data, layer).axes == ("t", "z", "y", "x")


def test_metadata_that_does_not_fit_the_array_is_ignored():
    # Stale labels from another array must not be applied to this one.
    data = np.zeros((4, 6))
    layer = FakeLayer(data, metadata={_axes.METADATA_KEY: ("z", "y", "x")})
    guess = _axes.resolve(data, layer)
    assert guess.axes == (None, None)
    assert not guess.declared


def test_stored_placeholders_are_not_read_back_as_a_declaration():
    # An earlier session may have written napari's own numbering onto the
    # layer. Read back as names, it would masquerade as somebody's answer.
    data = np.zeros((4, 6))
    layer = FakeLayer(data, metadata={_axes.METADATA_KEY: ("-2", "-1")})
    guess = _axes.resolve(data, layer)
    assert guess.axes == (None, None)
    assert not guess.declared


# -- rung 3: napari's own layer state --------------------------------------


def test_layer_axis_labels_are_used_when_they_are_real():
    data = np.zeros((5, 4, 6))
    layer = FakeLayer(data, axis_labels=("depth", "row", "col"))
    guess = _axes.resolve(data, layer)
    assert guess.axes == ("depth", "row", "col")
    assert guess.declared


def test_naparis_placeholder_labels_are_not_axis_names():
    # napari fills unlabelled axes with '-3', '-2', '-1'. Reading those as
    # names would produce three axes called minus-something.
    data = np.zeros((5, 4, 6))
    layer = FakeLayer(data, axis_labels=("-3", "-2", "-1"))
    guess = _axes.resolve(data, layer)
    assert guess.axes == (None, None, None)
    assert not guess.declared


def test_a_layer_may_name_only_some_of_its_axes():
    data = np.zeros((5, 4, 6))
    layer = FakeLayer(data, axis_labels=("lifetime", "-2", "-1"))
    guess = _axes.resolve(data, layer)
    assert guess.axes == ("lifetime", None, None)
    assert not guess.declared


def test_rgb_settles_a_trailing_channel_axis():
    data = np.zeros((4, 6, 3))
    layer = FakeLayer(data, axis_labels=("row", "col", "-1"), rgb=True)
    guess = _axes.resolve(data, layer)
    assert guess.axes == ("row", "col", "c")
    assert guess.declared


# -- what is left unnamed --------------------------------------------------


def test_an_unnamed_plane_stays_unnamed():
    # Nothing here says which axis is y. Calling one of them y anyway would be
    # a claim about the data, and napari has a perfectly good name for it.
    guess = _axes.resolve(np.zeros((4, 6)))
    assert guess.axes == (None, None)
    assert not guess.declared
    assert _axes.display(guess.axes) == ("-2", "-1")


def test_an_unnamed_stack_stays_unnamed():
    guess = _axes.resolve(np.zeros((5, 4, 6)))
    assert guess.axes == (None, None, None)
    assert str(guess) == "-3 -2 -1"


def test_a_trailing_extent_of_three_is_still_read_as_rgb():
    # The one name worth inferring: c is the axis ops declare as optional, and
    # skop fills an optional slot by name only. Unnamed, an RGB axis gets
    # iterated over three times instead of handed to to_gray whole.
    guess = _axes.resolve(np.zeros((4, 6, 3)))
    assert guess.axes == (None, None, "c")
    assert not guess.declared


def test_a_displayed_trailing_axis_is_spatial_not_rgb():
    # A (4, 6, 3) volume being shown as x=3 is genuinely three planes wide;
    # calling it RGB would transpose the image out from under the user.
    data = np.zeros((4, 6, 3))
    viewer = FakeViewer(displayed=(1, 2))
    assert _axes.resolve(data, None, viewer).axes == (None, None, None)


def test_nothing_else_is_ever_invented():
    # The hazard this avoids: a guessed name is *consumed* by the slot of that
    # name rather than iterated over. Only c is inferred, and only from a
    # trailing extent, which reads a real property of the data.
    for ndim in (4, 5, 6):
        assert _axes.resolve(np.zeros((2,) * ndim)).axes == (None,) * ndim


# -- what the viewer is displaying -----------------------------------------


def test_the_displayed_plane_is_reported_innermost_last():
    assert _axes.displayed(FakeViewer(displayed=(0, 1)), 3) == (0, 1)
    assert _axes.displayed(FakeViewer(displayed=(2, 1)), 3) == (2, 1)


def test_a_3d_view_is_read_too():
    # Rung four only looked in 2-D view, because it was after a plane. What is
    # on screen matters just as much when three axes are being rendered.
    assert _axes.displayed(FakeViewer(displayed=(0, 1, 2)), 3) == (0, 1, 2)


def test_the_viewers_axes_are_right_aligned_onto_the_layers():
    # napari's dims belong to the viewer: a 3-D layer in a 4-D viewer has the
    # viewer's axis 1 as its own axis 0.
    viewer = FakeViewer(displayed=(2, 3), ndim=4)
    assert _axes.displayed(viewer, 3) == (1, 2)


def test_a_viewer_showing_axes_this_layer_lacks_says_nothing():
    # The viewer is displaying two of the outer axes, which a 2-D layer
    # right-aligned into that space does not have at all.
    assert _axes.displayed(FakeViewer(displayed=(0, 1), ndim=4), 2) == ()
    assert _axes.displayed(FakeViewer(), 3) == ()
    assert _axes.displayed(None, 3) == ()


# -- odds and ends ---------------------------------------------------------


def test_labels_may_be_typed_with_spaces_or_commas():
    assert _axes.parse(" z, y ,x ") == ("z", "y", "x")
    assert _axes.parse("lifetime y x") == ("lifetime", "y", "x")
    assert _axes.parse("   ") == ()


def test_a_placeholder_typed_back_is_still_a_placeholder():
    # Someone editing '-3 -2 -1' into '-3 -2 c' has named one axis, not three.
    assert _axes.parse("-3 -2 c") == (None, None, "c")


def test_a_partial_answer_is_not_written_back_to_the_layer():
    layer = FakeLayer(np.zeros((4, 4)))
    _axes.remember(layer, ("y", None))
    assert _axes.METADATA_KEY not in layer.metadata


def test_the_layer_is_found_from_the_data_the_combo_handed_over():
    # napari's ImageData annotation drops the layer and keeps only its array,
    # and the array is the one thing that identifies it again.
    data = np.zeros((4, 4))
    layer = FakeLayer(data)
    viewer = FakeViewer(layers=[FakeLayer(np.ones((4, 4))), layer])
    assert _axes.layer_for(viewer, data) is layer
    assert _axes.layer_for(viewer, np.zeros((4, 4))) is None
    assert _axes.layer_for(None, data) is None


def test_slider_positions_are_reported_per_axis_index():
    # This is what makes "the current Z slice" mean the one being looked at.
    # By index, because an axis nobody has named still has a slider.
    viewer = FakeViewer(current_step=(7, 0, 0))
    assert _axes.positions((None, None, None), viewer) == {0: 7, 1: 0, 2: 0}
    # The viewer's sliders are right-aligned onto this layer's axes, and a
    # viewer with fewer of them than the array says nothing at all.
    assert _axes.positions(("y", "x"), viewer) == {0: 0, 1: 0}
    assert _axes.positions(("z", "y", "x"), FakeViewer(current_step=(0, 0))) == {}
