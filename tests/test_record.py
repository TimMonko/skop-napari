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
    assert "image" in panel._roots.value


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
    assert "scale" in panel._table[0][0].value


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
    panel._recording.value = False

    assert panel._session.executor is panel._session.runner
    panel._session.executor.run(scale, {"image": image.data})

    assert panel._recorder.graph().steps() == []


def test_turning_recording_on_declares_what_was_opened_meanwhile(viewer, panel):
    panel._recording.value = False
    late = _add_image(viewer, "late")
    assert SKOP not in late.metadata

    panel._recording.value = True

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
    assert len(panel._table) == 1
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

    result = session.executor.run(
        sharpen,
        {"image": image.data},
        on_progress=print,
        on_start=print,
    )

    (fn, args, options) = noting.calls[0]
    assert fn is sharpen
    assert args["image"] is image.data
    # What the caller asked of the runner reaches it: progress and Cancel.
    assert options == {"on_progress": print, "on_start": print}
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
    assert "scale" in panel._status.value and "offset" in panel._status.value


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
        assert "scale" in record._table[0][0].value
    finally:
        record._session.runner.close()


def test_one_session_per_viewer(viewer):
    assert session_for(viewer) is session_for(viewer)
