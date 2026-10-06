"""The Record panel, against a real napari viewer.

The ops here are declared with scikit-ops' ``@op()`` and name no environment,
so the panel runs them in this interpreter: no Appose, nothing to build, and
the file costs a fraction of a second. That is the contract the recorder reads
(decision 44); the last two tests are about the kinds of op it refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop import OpSpec, is_op, op
from skop import spec as spec_of
from skop_napari import RecordPanel
from skop_napari._host import SKOP, UnknownLayer


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


@op(env="demo")
def sharpen(image: np.ndarray) -> np.ndarray:
    """Names an environment, so the panel will not run it here."""
    return image


def _spec(fn) -> OpSpec:
    return spec_of(fn)


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
    """A panel over two recordable ops, running in this interpreter."""
    specs = [_spec(scale), _spec(offset)]
    widget = RecordPanel(
        viewer,
        specs=specs,
        registry={spec.name: fn for spec, fn in zip(specs, (scale, offset))},
    )
    viewer.window.add_dock_widget(widget)
    return widget


def _add_image(viewer, name="image"):
    return viewer.add_image(np.ones((8, 8), dtype="float32"), name=name)


def _choose(panel, fn) -> None:
    """Pick an op by the function that declares it."""
    wanted = _spec(fn).name
    panel._picker.value = next(
        label for label, spec in panel._by_label.items() if spec.name == wanted
    )


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


def test_running_an_op_records_it_and_draws_its_layer(viewer, panel):
    image = _add_image(viewer)
    _choose(panel, scale)

    panel._start()

    run = panel._recorder.graph().current("scale")
    assert run is not None
    assert run.op == _spec(scale).name
    assert run.params == {"factor": 2.0}
    assert run.inputs == {"image": _root_handle(image)}
    assert run.outputs  # the store has the result, under a handle of its own

    drawn = viewer.layers["scale"]
    assert drawn.metadata[SKOP]["step"] == "scale"
    assert np.allclose(drawn.data, image.data * 2)
    assert "scale" in panel._status.value


def test_the_same_op_twice_adds_a_run_rather_than_a_step(viewer, panel):
    _add_image(viewer)
    _choose(panel, scale)

    panel._start()
    panel._start()

    graph = panel._recorder.graph()
    assert graph.steps() == ["scale"]
    assert len(graph.runs_of("scale")) == 2
    # Two layers would mean a step re-run beside itself instead of in place.
    assert len([lbl for lbl in viewer.layers if lbl.name == "scale"]) == 1


def test_an_edit_is_marked_and_reverting_puts_the_run_back(viewer, panel):
    _add_image(viewer)
    _choose(panel, scale)
    panel._start()
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


def test_produce_brings_the_closure_of_a_target_up_to_date(viewer, panel):
    image = _add_image(viewer)
    _choose(panel, scale)
    panel._start()
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
    panel._produce_()

    after = panel._recorder.graph()
    assert not after.is_stale("scale")
    assert not after.is_stale("offset")
    assert len(after.runs_of("scale")) == 2
    # And the layer shows the data that moved, not the data that was there.
    assert np.allclose(viewer.layers["scale"].data, image.data * 2)
    assert "scale" in panel._status.value and "offset" in panel._status.value


def test_the_ops_are_declared_the_way_scikit_ops_asks():
    """Decision 44: the log speaks scikit-ops' contract, and only that one."""
    assert is_op(scale) and is_op(offset) and is_op(sharpen)
    assert not is_op(lambda x: x)
    assert _spec(scale).is_workflow  # no env, so it runs on the host
    assert not _spec(sharpen).is_workflow


def test_an_op_that_names_an_environment_is_refused_in_words(viewer):
    """The recorder can read it but cannot run it where it asked to run.

    The panel's runner calls the function here; an op that names an environment
    belongs in a scikit-ops worker, which is not wired to the recorder yet. So
    the op is offered and explained rather than run in the wrong place.
    """
    spec = _spec(sharpen)
    widget = RecordPanel(viewer, specs=[spec], registry={spec.name: sharpen})
    widget._picker.value = next(iter(widget._by_label))

    refusal = widget._refusal(widget.spec)
    assert refusal is not None
    assert "environment" in refusal
    assert not widget._button.enabled


def test_a_function_that_is_not_an_op_is_refused_in_words(viewer):
    """A plain function carries no marker, so it is not an op to record."""
    spec = _spec(scale)
    widget = RecordPanel(
        viewer,
        specs=[spec],
        # A function that looks runnable but was never declared an op.
        registry={spec.name: object()},
    )
    widget._picker.value = next(iter(widget._by_label))

    refusal = widget._refusal(widget.spec)
    assert refusal is not None
    assert "not declared" in refusal
    assert not widget._button.enabled
