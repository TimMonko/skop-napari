"""Turning what the viewer is displaying into which axis fills which slot.

The rest of the row is exercised against a real viewer in test_panel; this is
the arithmetic underneath it, which is easier to corner here.
"""

from __future__ import annotations

from opspec.op import Slot

from skop_napari._plans import _guided


def slots(*names):
    """Slots as an op declares them, a trailing '?' marking one optional."""
    return tuple(
        Slot(name.rstrip("?") or None, optional=name.endswith("?")) for name in names
    )


def test_the_displayed_axes_fill_the_spatial_slots():
    yx = slots("y", "x")
    assert _guided((1, 2), yx, (0, 1), 3) == (0, 1)
    assert _guided((1, 2), yx, (2, 0), 3) == (2, 0)


def test_a_2d_op_in_a_3d_view_takes_the_innermost_two():
    assert _guided((1, 2), slots("y", "x"), (0, 1, 2), 3) == (1, 2)


def test_a_3d_op_in_a_2d_view_keeps_its_z_from_the_names():
    # Right-aligned from the other side: only y and x are on screen, so only y
    # and x are decided here, and z stays wherever it already was.
    assert _guided((0, 1, 2), slots("z", "y", "x"), (2, 1), 3) == (0, 2, 1)


def test_a_slot_that_is_not_spatial_is_left_alone():
    # c is about what the data means, which layout says nothing about.
    assert _guided((1, 2, 0), slots("y", "x", "c?"), (1, 2), 3) == (1, 2, 0)


def test_an_optional_slot_losing_its_axis_goes_empty():
    # Filling c by position is exactly what skop design 0006 forbids: an op
    # would average across a timepoint axis rather than iterate over it.
    assert _guided((0, 1, 2), slots("y", "x", "c?"), (1, 2), 3) == (1, 2, None)


def test_a_required_slot_losing_its_axis_takes_a_free_one():
    assert _guided((0, 1, 2, 3), slots("z", "y", "x", "c"), (2, 3), 4) == (0, 2, 3, 1)


def test_a_viewer_with_nothing_to_say_changes_nothing():
    assert _guided((1, 2), slots("y", "x"), (), 3) == (1, 2)


def test_a_wildcard_slot_is_not_spatial():
    assert _guided((0, 1), slots("*", "*"), (1, 2), 3) == (0, 1)
