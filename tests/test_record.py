"""Recording, against a real napari viewer.

The Record panel runs nothing: the Ops panel runs ops, and a session's
listening runner writes them down on the way past. These tests run the ops
here the way the Ops panel does -- through ``session.executor`` -- and the ops
are declared with scikit-ops' ``@op()`` and name no environment, so they run in
this interpreter: no Appose, nothing to build.

What an op that *does* name an environment gets is tested with a runner that
only notes what it was asked to do. Building a real environment is the Ops
panel's tests' business; what matters here is that the call reaches the runner
rather than being refused, or run in the wrong place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import numpy as np
import pytest
from qtpy.QtCore import Qt

import skop
from skop import Choices, OpSpec, op
from skop_napari import OpsPanel, RecordPanel
from skop_napari._host import SKOP, UnknownLayer
from skop_napari._session import session_for


@op()
def scale(image: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Multiply every pixel.

    Args:
        image: The picture to scale.
        factor: What to multiply it by.
    """
    return image * factor


@op()
def offset(image: np.ndarray, shift: float = 10.0) -> np.ndarray:
    """Add a constant to every pixel.

    Args:
        image: The picture to shift.
        shift: What to add to it.
    """
    return image + shift


@op()
def scale_then_offset(image: np.ndarray) -> np.ndarray:
    """A workflow: calls two ops through skop.run, rather than being one."""
    return skop.run(offset, {"image": skop.run(scale, {"image": image})})


@op()
def either_then_offset(
    image: np.ndarray,
    first: Annotated[Callable, Choices(scale=scale, offset=offset)] = scale,
) -> np.ndarray:
    """A workflow handed an op: which one to run first is a parameter."""
    return skop.run(offset, {"image": skop.run(first, {"image": image})})


@op(env="demo")
def sharpen(image: np.ndarray) -> np.ndarray:
    """Names an environment, so it must reach a runner that can honour it."""
    return image


def _spec(fn) -> OpSpec:
    return OpSpec.from_op(fn)


OPS = (scale, offset, scale_then_offset, either_then_offset, sharpen)


class NotingRunner:
    """A runner that notes what it was asked and answers without running it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def run(self, fn, args=None, **options):
        self.calls.append((fn, dict(args or {}), options))
        return (args or {})["image"] + 1


@pytest.fixture
def viewer(qtbot):
    """A viewer of our own, rather than napari's fixture.

    ``make_napari_viewer`` reads ``request.node.rep_call`` while it tears the
    viewer down, and nothing in the installed napari package sets that -- the
    hook that does lives in napari's own repository conftest. Without it a
    test passes and then errors in teardown, so the viewer is made here
    instead, which is the same viewer minus the hook. ``qtbot`` is asked for
    only to guarantee a Qt application exists first.
    """
    from napari import Viewer

    made = Viewer(show=False)
    yield made
    made.close()


@pytest.fixture
def panel(viewer):
    """A Record panel over a session that knows the test ops."""
    specs = [_spec(fn) for fn in OPS]
    widget = RecordPanel(
        viewer,
        specs=specs,
        registry={spec.name: fn for spec, fn in zip(specs, OPS)},
    )
    viewer.window.add_dock_widget(widget)
    yield widget
    widget._session.runner.close()


def _add_image(viewer, name="image"):
    return viewer.add_image(np.ones((8, 8), dtype="float32"), name=name)


def _run(panel, fn, layer, **params):
    """Run an op the way the Ops panel does, and draw what came back.

    The same three steps: call the executor with the layer's data, take the
    step it was recorded as, and have the session draw the output.
    """
    session = panel._session
    steps: list[str] = []
    result = session.executor.run(
        fn, {"image": layer.data, **params}, on_recorded=steps.append
    )
    session.draw(steps[0], "result", result, "image", {"name": steps[0]})
    return steps[0]


def _root_handle(layer) -> str:
    return layer.metadata[SKOP]["handle"]


def test_a_layer_in_the_viewer_becomes_a_root(viewer, panel):
    image = _add_image(viewer)

    # No declaration step: whatever the viewer holds, and this host has not
    # seen, is an input. The store knows the array by identity, which is what
    # ties a layer to the data a run was recorded reading.
    handle = _root_handle(image)
    assert panel._recorder.store.handle_of(image.data) == handle
    assert panel._recorder.store.get(handle).shape == (8, 8)
    assert np.array_equal(panel._host.array_for(image), image.data)
    assert "image" in panel._input_layers


def test_the_panel_runs_nothing_itself(panel):
    # What this panel was: a picker, inputs, and a button that called the op.
    assert not hasattr(panel, "_picker")
    assert not hasattr(panel, "_start")


def test_an_op_run_elsewhere_is_recorded_and_draws_its_layer(viewer, panel):
    image = _add_image(viewer)

    step = _run(panel, scale, image)

    assert step == "scale"
    run = panel._recorder.graph().current("scale")
    assert run is not None
    assert run.op == _spec(scale).name
    assert run.params == {}
    assert run.inputs == {"image": _root_handle(image)}
    assert run.outputs

    drawn = viewer.layers["scale"]
    assert drawn.metadata[SKOP]["step"] == "scale"
    assert np.allclose(drawn.data, image.data * 2)
    # The table is redrawn because the session said so, not because the panel
    # was asked: nothing here called its refresh.
    assert panel._table.item(0, 1).text() == "scale"


def test_the_same_op_twice_adds_a_run_rather_than_a_step(viewer, panel):
    image = _add_image(viewer)

    _run(panel, scale, image)
    _run(panel, scale, image, factor=3.0)

    graph = panel._recorder.graph()
    assert graph.steps() == ["scale"]
    assert len(graph.runs_of("scale")) == 2
    # Two layers would mean a step re-run beside itself instead of in place.
    assert len([lbl for lbl in viewer.layers if lbl.name == "scale"]) == 1
    assert np.allclose(viewer.layers["scale"].data, image.data * 3)


def test_nothing_is_recorded_while_recording_is_off(viewer, panel):
    image = _add_image(viewer)
    panel._recording.setChecked(False)

    assert panel._session.executor is panel._session.runner
    panel._session.executor.run(scale, {"image": image.data})

    assert panel._recorder.graph().steps() == []


def test_turning_recording_on_declares_what_was_opened_meanwhile(viewer, panel):
    panel._recording.setChecked(False)
    late = _add_image(viewer, "late")
    assert SKOP not in late.metadata

    panel._recording.setChecked(True)

    assert late.metadata[SKOP]["step"] is None


def test_a_workflow_is_one_step_and_its_ops_are_its_detail(viewer, panel):
    image = _add_image(viewer)

    step = _run(panel, scale_then_offset, image)

    graph = panel._recorder.graph()
    assert step == "scale_then_offset"
    assert graph.steps() == ["scale_then_offset"]
    parent = graph.current("scale_then_offset")
    assert [c.step for c in graph.children_of(parent)] == [
        "scale_then_offset.scale",
        "scale_then_offset.offset",
    ]
    # One row, and one layer: what the workflow did on the way is not drawn.
    assert panel._table.rowCount() == 1
    assert [lyr.name for lyr in viewer.layers] == ["image", "scale_then_offset"]
    assert np.allclose(viewer.layers["scale_then_offset"].data, image.data * 2 + 10)


def test_a_workflow_handed_an_op_records_it_by_id_and_can_run_again(viewer, panel):
    """The crash: a function-valued parameter is not JSON, so it was refused."""
    image = _add_image(viewer)

    step = _run(panel, either_then_offset, image, first=offset)

    graph = panel._recorder.graph()
    assert graph.current(step).params == {"first": _spec(offset).name}
    # An ID goes back to the op it names, from the parameter's own choices.
    panel._recorder.rerun(either_then_offset, step)
    assert len(panel._recorder.graph().runs_of(step)) == 2


def test_an_op_that_names_an_environment_reaches_the_runner(viewer):
    """It is run where it asked to be, not refused and not run here instead."""
    noting = NotingRunner()
    spec = _spec(sharpen)
    widget = RecordPanel(viewer, specs=[spec], registry={spec.name: sharpen}, root=None)
    session = widget._session
    # The session's recorder was built over the real runner; point it at the
    # noting one, which is all a different runner is to the recorder.
    session.recorder.inner = noting
    image = _add_image(viewer)

    started = []
    result = session.executor.run(
        sharpen,
        {"image": image.data},
        on_progress=print,
        on_start=started.append,
    )

    (fn, args, options) = noting.calls[0]
    assert fn is sharpen
    assert args["image"] is image.data
    # What the caller asked of the runner reaches it: progress, and a cancel
    # handle -- watched on the way, so a cancelled run is recorded as one.
    assert set(options) == {"on_progress", "on_start"}
    assert options["on_progress"] is print

    class Task:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    task = Task()
    options["on_start"](task)
    (handed,) = started
    handed.cancel()
    assert task.cancelled
    assert np.array_equal(result, image.data + 1)
    assert session.recorder.graph().current("sharpen") is not None
    session.runner.close()


def test_an_edit_is_marked_and_reverting_puts_the_run_back(viewer, panel):
    image = _add_image(viewer)
    _run(panel, scale, image)
    drawn = viewer.layers["scale"]
    recorded = drawn.data.copy()

    drawn.data = drawn.data + 1  # the user paints on the result

    assert panel._recorder.graph().is_edited("scale")
    # Until they say which they want, nothing may read the layer: handing the
    # recorded array back would quietly throw the edit away.
    with pytest.raises(UnknownLayer):
        panel._host.array_for(drawn)

    panel._revert("scale")

    assert not panel._recorder.graph().is_edited("scale")
    assert np.array_equal(drawn.data, recorded)


def test_produce_brings_the_closure_of_a_target_up_to_date(viewer, panel, qtbot):
    image = _add_image(viewer)
    _run(panel, scale, image)
    # The second op reads what the first drew, as its layer.
    panel._host.run("offset", _spec(offset).name, {"image": viewer.layers["scale"]})

    image.data = image.data + 1  # the root moves under both of them

    before = panel._recorder.graph()
    assert before.is_stale("scale")
    # offset is not stale on its own account: what it read is still what scale
    # last produced. It is the closure that reaches past it to scale, and
    # running scale is what then makes offset stale in turn.
    assert set(before.closure(["offset"])) == {"scale", "offset"}

    panel._tickle("offset", True)
    # Produced on a worker, and drawn when it returns.
    panel._produce_()
    assert panel._producing is not None
    qtbot.waitUntil(lambda: panel._producing is None, timeout=30_000)

    after = panel._recorder.graph()
    assert not after.is_stale("scale")
    assert not after.is_stale("offset")
    assert len(after.runs_of("scale")) == 2
    # And the layer shows the data that moved, not the data that was there.
    assert np.allclose(viewer.layers["scale"].data, image.data * 2)
    assert "scale" in panel._status.text() and "offset" in panel._status.text()


def test_the_ops_panel_records_through_the_session(viewer, qtbot):
    """The whole path: a real panel, a worker thread, and a listening runner."""
    specs = [_spec(scale)]
    record = RecordPanel(viewer, specs=specs, registry={specs[0].name: scale})
    ops = OpsPanel(viewer)
    try:
        assert ops._session is record._session
        assert ops._runner is record._session.runner
        image = _add_image(viewer)
        ops._by_label = {"scale": specs[0]}
        ops._picker.choices = ["scale"]
        ops._picker.value = "scale"
        # The combo is built with the layer present, so it already holds it:
        # picking one is the Ops panel's job and not what is under test.
        combo = next(w for w in ops._inputs.widgets if w.name == "image")
        assert combo.value is image.data

        with qtbot.waitSignal(ops.finished, timeout=60_000):
            ops._start()

        run = record._recorder.graph().current("scale")
        assert run is not None
        assert run.inputs == {"image": _root_handle(image)}
        # The panel names its layers its own way; the link is what says which
        # step one draws.
        drawn = record._host.display.layer_for("scale")
        assert drawn is not None and drawn is not image
        assert np.allclose(drawn.data, image.data * 2)
        assert record._table.item(0, 1).text() == "scale"
    finally:
        record._session.runner.close()


def test_one_session_per_viewer(viewer):
    assert session_for(viewer) is session_for(viewer)


# -- the panel, as a place to read a session -------------------------------


def _two_steps(viewer, panel):
    """scale, then offset reading what scale drew: a path with a middle."""
    image = _add_image(viewer)
    _run(panel, scale, image)
    panel._host.run("offset", _spec(offset).name, {"image": viewer.layers["scale"]})
    return image


def _row(panel, step):
    return next(
        row
        for row in range(panel._table.rowCount())
        if panel._table.item(row, 1).text() == step
    )


def _tint(panel, step):
    """The alpha a row is tinted with, or 0 when it is not tinted at all."""
    brush = panel._table.item(_row(panel, step), 1).background()
    return brush.color().alpha() if brush.style() != Qt.BrushStyle.NoBrush else 0


def _badges(panel):
    return {n.step: n.badge for n in panel._host.view().nodes}


def test_the_table_says_what_a_step_read_made_and_ran_with(viewer, panel):
    image = _add_image(viewer)
    _run(panel, scale, image, factor=3.0)
    _run(panel, scale, image, factor=4.0)

    row = _row(panel, "scale")
    cell = lambda column: panel._table.item(row, column).text()
    assert cell(3) == "image ← image"
    assert cell(4) == "result"
    assert cell(5) == "factor=4.0"
    assert cell(6) == "2"
    assert "fresh" in cell(2)


def test_choosing_another_layer_for_an_input_makes_its_readers_stale(
    viewer, panel, qtbot
):
    first = _add_image(viewer)
    _run(panel, scale, first)
    other = viewer.add_image(np.full((8, 8), 5, dtype="float32"), name="other")
    assert _badges(panel)["scale"] == "fresh"

    # What the dropdown does when someone picks the other layer.
    choices = panel._input_layers["image"]
    assert first in choices and other in choices
    panel._input_chosen("image", choices.index(other))

    assert panel._host.layer_of("image") is other
    assert panel._recorder.graph().is_stale("scale")
    assert _badges(panel)["scale"] == "stale"

    panel._tickle("scale", True)
    panel._produce_()
    qtbot.waitUntil(lambda: panel._producing is None, timeout=30_000)

    # Produced from the new layer, which is what choosing it asked for.
    assert not panel._recorder.graph().is_stale("scale")
    assert np.allclose(viewer.layers["scale"].data, other.data * 2)


def test_a_root_keeps_its_name_when_its_layer_is_renamed(viewer, panel):
    image = _add_image(viewer)
    _run(panel, scale, image)

    image.name = "renamed"
    image.data = image.data + 1  # then the pixels move

    # Under the old name the readers know, not under the new one: otherwise
    # the move would be a root nobody read, and nothing would go stale.
    assert image.metadata[SKOP]["root"] == "image"
    assert panel._recorder.graph().is_stale("scale")


def test_a_layer_displaced_from_an_input_does_not_move_it_again(viewer, panel):
    first = _add_image(viewer)
    _run(panel, scale, first)
    other = viewer.add_image(np.full((8, 8), 5, dtype="float32"), name="other")
    panel._input_chosen("image", panel._input_layers["image"].index(other))
    versions = len(panel._recorder.graph().roots_of("image"))

    first.data = first.data + 1

    # It is its own input now, not a second source for the name.
    assert len(panel._recorder.graph().roots_of("image")) == versions
    assert first.metadata[SKOP]["root"] == "image (2)"


def test_removing_an_inputs_layer_marks_its_readers_missing_not_stale(viewer, panel):
    image = _add_image(viewer)
    _run(panel, scale, image)

    viewer.layers.remove(image)

    graph = panel._recorder.graph()
    assert graph.gone == {"image"}
    assert not graph.is_stale("scale")
    assert _badges(panel)["scale"] == "missing"
    assert "input gone" in panel._table.item(_row(panel, "scale"), 2).text()
    # The dropdown says so, rather than quietly offering another layer.
    assert panel._input_layers["image"][0] is None

    # And the name can be filled again, which is what undoes it.
    back = _add_image(viewer, "back")
    panel._input_chosen("image", panel._input_layers["image"].index(back))
    assert _badges(panel)["scale"] == "stale"


def test_a_layer_changed_while_nothing_recorded_is_found_on_the_next_look(
    viewer, panel
):
    image = _add_image(viewer)
    _run(panel, scale, image)
    panel._recording.setChecked(False)

    image.data = image.data + 1
    assert _badges(panel)["scale"] == "fresh"  # nobody was listening

    panel._recording.setChecked(True)
    assert _badges(panel)["scale"] == "stale"


def test_painting_a_result_marks_it_edited(viewer, panel):
    image = _add_image(viewer)
    session = panel._session
    steps: list[str] = []
    result = session.executor.run(
        scale, {"image": image.data}, on_recorded=steps.append
    )
    labels = session.draw(
        steps[0], "result", result.astype("int32"), "labels", {"name": "painted"}
    )
    assert _badges(panel)["scale"] == "fresh"

    labels.paint((2, 2), 7)  # in place, and reported as paint rather than data

    assert _badges(panel)["scale"] == "edited"


def test_painting_an_input_stores_a_new_version_once_the_strokes_stop(
    viewer, panel, qtbot
):
    session = panel._session
    mask = viewer.add_labels(np.ones((8, 8), dtype="int32"), name="mask")
    steps: list[str] = []
    session.executor.run(scale, {"image": mask.data}, on_recorded=steps.append)
    before = len(panel._recorder.graph().roots_of("mask"))

    for coord in ((1, 1), (1, 2), (1, 3)):
        mask.paint(coord, 5)

    qtbot.waitUntil(
        lambda: len(panel._recorder.graph().roots_of("mask")) > before,
        timeout=5_000,
    )
    # Three strokes, one version.
    assert len(panel._recorder.graph().roots_of("mask")) == before + 1
    assert panel._recorder.graph().is_stale("scale")


def test_pointing_at_a_step_lights_the_path_through_it(viewer, panel):
    _two_steps(viewer, panel)

    panel.select("offset")

    # What it was made from is lit, back to the input; nothing else is.
    assert panel._canvas.lit == {("root:image", "scale"), ("scale", "offset")}
    assert _tint(panel, "scale") == 60
    assert _tint(panel, "offset") == 0
    # And the layer is found in the viewer.
    assert viewer.layers.selection.active is viewer.layers["offset"]
    assert "image" in panel._detail.text() and "scale" in panel._detail.text()

    panel.select("scale")

    # From here the path runs both ways: in from the input, out to offset.
    assert panel._canvas.lit == {("root:image", "scale"), ("scale", "offset")}
    assert (
        panel._table.item(_row(panel, "offset"), 1).background().color().alpha() == 55
    )
    assert "feeds: offset" in panel._detail.text()


def test_clicking_a_node_in_the_graph_selects_it(viewer, panel):
    _two_steps(viewer, panel)

    panel._canvas.picked.emit("scale")
    assert panel._selected == "scale"
    assert panel._table.selectionModel().selectedRows()[0].row() == _row(panel, "scale")

    # A root is not a step, so pointing at one points at nothing.
    panel._canvas.picked.emit("root:image")
    assert panel._selected is None


def test_picking_a_layer_in_the_viewer_points_at_its_step(viewer, panel):
    _two_steps(viewer, panel)

    viewer.layers.selection.active = viewer.layers["scale"]

    assert panel._selected == "scale"
    assert ("scale", "offset") in panel._canvas.lit


def test_produce_draws_a_deleted_result_again_without_running_it(viewer, panel, qtbot):
    image = _add_image(viewer)
    _run(panel, scale, image)
    viewer.layers.remove(viewer.layers["scale"])
    assert "no layer" in panel._table.item(_row(panel, "scale"), 2).text()

    panel._tickle("scale", True)
    panel._produce_()
    qtbot.waitUntil(lambda: panel._producing is None, timeout=30_000)

    # Back, from the store: the output was never lost, so nothing ran.
    assert np.allclose(viewer.layers["scale"].data, image.data * 2)
    assert len(panel._recorder.graph().runs_of("scale")) == 1
    assert "again" in panel._status.text()
    assert "no layer" not in panel._table.item(_row(panel, "scale"), 2).text()


def test_the_same_op_on_another_input_is_a_step_of_its_own(viewer, panel):
    first = _add_image(viewer)
    second = viewer.add_image(np.full((8, 8), 3, dtype="float32"), name="second")

    _run(panel, scale, first)
    _run(panel, scale, second)
    _run(panel, scale, second, factor=5.0)

    graph = panel._recorder.graph()
    assert graph.steps() == ["scale", "scale_2"]
    assert len(graph.runs_of("scale_2")) == 2
    # Two results, side by side, rather than one overwriting the other.
    assert np.allclose(viewer.layers["scale"].data, first.data * 2)
    assert np.allclose(viewer.layers["scale_2"].data, second.data * 5)


def _wheel(canvas, delta, modifiers):
    from qtpy.QtCore import QPoint, QPointF
    from qtpy.QtGui import QWheelEvent

    centre = QPointF(canvas.viewport().rect().center())
    return QWheelEvent(
        centre,
        QPointF(canvas.viewport().mapToGlobal(centre.toPoint())),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def test_ctrl_wheel_zooms_the_graph_and_a_plain_wheel_is_left_to_the_panel(
    viewer, panel
):
    _two_steps(viewer, panel)
    canvas = panel._canvas
    before = canvas.transform().m11()

    plain = _wheel(canvas, 120, Qt.KeyboardModifier.NoModifier)
    canvas.wheelEvent(plain)
    assert not plain.isAccepted()  # passed up, so the panel scrolls
    assert canvas.transform().m11() == before

    ctrl = _wheel(canvas, 120, Qt.KeyboardModifier.ControlModifier)
    canvas.wheelEvent(ctrl)
    assert ctrl.isAccepted()
    assert canvas.transform().m11() > before


def test_a_zoom_survives_a_redraw_until_asked_to_fit(viewer, panel):
    _two_steps(viewer, panel)
    canvas = panel._canvas
    fitted = canvas.transform().m11()

    canvas.zoom(2.0)
    zoomed = canvas.transform().m11()
    panel.select("scale")  # redraws the graph

    assert canvas.transform().m11() == pytest.approx(zoomed)
    canvas.fit()
    # Not exact: the selected node's heavier border widens what is fitted.
    assert canvas.transform().m11() == pytest.approx(fitted, rel=0.02)


def test_dragging_the_graph_pans_it_rather_than_selecting(viewer, panel, qtbot):
    _two_steps(viewer, panel)
    panel.select("scale")
    canvas = panel._canvas
    canvas.zoom(3.0)
    start = canvas.viewport().rect().center()

    qtbot.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    qtbot.mouseMove(canvas.viewport(), start + start / 4)
    qtbot.mouseRelease(
        canvas.viewport(), Qt.MouseButton.LeftButton, pos=start + start / 4
    )

    assert panel._selected == "scale"


def test_keeping_an_edit_stores_it_and_stales_what_read_it(viewer, panel):
    image = _add_image(viewer)
    step = _run(panel, scale, image)
    after = _run(panel, offset, viewer.layers[step])
    result = viewer.layers[step]
    result.data = np.zeros_like(result.data)  # a hand edit, by assignment
    assert panel._recorder.graph().is_edited(step)

    panel.select(step)
    assert panel._keep_button.isVisibleTo(panel)
    panel._keep(step)
    graph = panel._recorder.graph()
    assert graph.is_accepted(step) and graph.is_stale(after)
    kept = panel._recorder.store.get(graph.accepted(step))
    assert not kept.any()
    # The kept layer can be run on, and what runs on it reads the kept pixels.
    again = _run(panel, offset, viewer.layers[step])
    assert again == after
    assert (viewer.layers[after].data == 10.0).all()
    assert panel._revert_button.isVisibleTo(panel)


def test_a_redrawn_box_output_is_laid_out_as_rectangles(viewer, panel):
    from skop import Role

    display = panel._session.display
    boxes = np.array([[1.0, 2.0, 5.0, 6.0], [0.0, 0.0, 3.0, 3.0]])
    layer = display.show(
        "found",
        boxes,
        step="found",
        role=Role.boxes,
        metadata={SKOP: {"id": "x", "step": "found", "output": "boxes"}},
    )
    assert type(layer).__name__ == "Shapes"
    assert len(layer.data) == 2
    assert set(layer.shape_type) == {"rectangle"}


def test_a_scalar_output_is_not_drawn(viewer, panel):
    display = panel._session.display
    count = len(viewer.layers)
    assert display.show("n", np.int64(3), step="n") is None
    assert len(viewer.layers) == count


def test_exporting_writes_the_closure_of_the_ticked_targets(viewer, panel, tmp_path):
    from skop_core import load

    image = _add_image(viewer)
    first = _run(panel, scale, image)
    second = _run(panel, offset, viewer.layers[first])
    _run(panel, offset, image)  # another branch, off the path
    panel._tickle(second, True)
    recipe = panel.export_recipe(tmp_path / "recipe.json")
    assert set(recipe["steps"]) == {first, second}
    assert load(tmp_path / "recipe.json") == recipe
    assert "Wrote 2 steps" in panel._status.text()


def test_exporting_with_nothing_ticked_takes_the_leaves(viewer, panel, tmp_path):
    image = _add_image(viewer)
    first = _run(panel, scale, image)
    second = _run(panel, offset, viewer.layers[first])
    recipe = panel.export_recipe(tmp_path / "recipe.json")
    assert recipe["targets"] == [second]
