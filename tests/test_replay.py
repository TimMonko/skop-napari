"""Replaying a recipe onto layers, against a real napari viewer.

The ops name no environment, so they run in this interpreter: nothing is
built. The recipes are recorded headlessly, the way a script or another
session would have made them, and replayed through the panel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skop import OpSpec, op
from skop.types import ImageOf, LabelsOf
from skop_core import InProcess, Recorder, SimpleStore, export, save
from skop_napari import RecordPanel, ReplayPanel
from skop_napari._host import SKOP


@op()
def scale(image: ImageOf[np.ndarray], factor: float = 2.0) -> ImageOf[np.ndarray]:
    """Multiply every pixel."""
    return np.asarray(image) * factor


@op()
def above(image: ImageOf[np.ndarray], level: float = 1.0) -> LabelsOf[np.ndarray]:
    """What lies above a level, as labels."""
    return (np.asarray(image) > level).astype(np.uint8)


@op()
def weighted(
    image: ImageOf[np.ndarray], weights: Path = Path("w.npy")
) -> ImageOf[np.ndarray]:
    """Reads a file named by a parameter: a path root in a recipe."""
    return np.asarray(image) * float(np.load(weights))


@op()
def stops(image: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]:
    """Fails, so a replay can fail partway."""
    raise RuntimeError("stops here")


#: Called from inside ``interrupted`` while it runs: a Cancel pressed mid-op.
WHILE_RUNNING: list = []


@op()
def interrupted(image: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]:
    """Lets a test press Cancel while it runs."""
    for press in WHILE_RUNNING:
        press()
    return np.asarray(image)


@op()
def picky(image: ImageOf[np.ndarray]) -> ImageOf[np.ndarray]:
    """Fails on bright images, so a batch can fail on one layer of several."""
    if np.asarray(image).max() > 100:
        raise ValueError("too bright")
    return np.asarray(image) * 3.0


OPS = (scale, above, weighted, stops, interrupted, picky)
REGISTRY = {OpSpec.from_op(fn).name: fn for fn in OPS}


def _recipe(tmp_path, *steps, targets=None):
    """A recipe recorded headlessly: scale, then above."""
    recorder = Recorder(SimpleStore(tmp_path / "recorded"), InProcess())
    image = np.ones((8, 8), dtype="float32")
    recorder.root("image", image)
    scaled = recorder.run(scale, "scaled", {"image": image, "factor": 3.0})
    recorder.run(above, "mask", {"image": scaled})
    for name, fn, args in steps:
        recorder.run(fn, name, {"image": scaled, **args})
    return export(recorder.graph(), targets or ["mask"], recorder.store)


@pytest.fixture
def viewer(qtbot):
    """A viewer of our own; see test_record.py for why not napari's."""
    from napari import Viewer

    made = Viewer(show=False)
    yield made
    made.close()


@pytest.fixture
def panel(viewer):
    specs = [OpSpec.from_op(fn) for fn in OPS]
    widget = ReplayPanel(viewer, specs=specs, registry=dict(REGISTRY))
    viewer.window.add_dock_widget(widget)
    yield widget
    widget._session.runner.close()


def _replay(panel, qtbot):
    with qtbot.waitSignal(panel.finished, timeout=60_000):
        panel.run()
    qtbot.wait(50)  # let the drawing queued on the GUI thread land


def test_nothing_runs_without_a_recipe(panel):
    assert panel.why_not() == "Open a recipe to replay."
    assert not panel._run.isEnabled()


def test_an_input_named_like_a_layer_is_bound_to_it(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="other")
    image = viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    assert panel._bindings == {"image": image}
    assert panel.why_not() is None


def test_the_only_layer_that_fits_is_bound(viewer, panel, tmp_path):
    only = viewer.add_image(np.ones((8, 8)), name="blobs")
    viewer.add_image(np.ones((4, 8, 8)), name="stack")  # wrong axes
    viewer.add_labels(np.zeros((8, 8), dtype=int), name="labels")  # wrong role
    panel.load(_recipe(tmp_path))
    (need,) = panel._prepared.roots
    assert panel.candidates(need) == [only]
    assert panel._bindings == {"image": only}


def test_a_choice_that_is_not_clear_is_left_to_the_user(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="a")
    viewer.add_image(np.ones((8, 8)), name="b")
    panel.load(_recipe(tmp_path))
    assert panel.why_not() == "Choose what stands for image."
    panel.bind("image", viewer.layers["b"])
    assert panel.why_not() is None


def test_a_replay_draws_its_target_and_records_it(viewer, panel, qtbot, tmp_path):
    viewer.add_image(np.full((8, 8), 0.5, dtype="float32"), name="blobs")
    save(_recipe(tmp_path), tmp_path / "recipe.json")
    panel.load(tmp_path / "recipe.json")
    _replay(panel, qtbot)

    mask = viewer.layers["mask"]
    assert type(mask).__name__ == "Labels"
    assert mask.data.all()  # 0.5 * 3 > 1
    assert "scaled" not in [layer.name for layer in viewer.layers]
    graph = panel._session.recorder.graph()
    assert graph.sources_of(graph.current("scaled")) == {"image": "root:blobs"}
    assert graph.current("scaled").params == {"factor": 3.0}
    assert mask.metadata[SKOP]["step"] == "mask"
    assert panel._status.text() == "Replayed 2 steps."


def test_every_step_can_be_shown(viewer, panel, qtbot, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    panel._every.setChecked(True)
    _replay(panel, qtbot)
    names = [layer.name for layer in viewer.layers]
    assert "scaled" in names and "mask" in names


def test_ticking_an_intermediate_produces_only_what_it_needs(
    viewer, panel, qtbot, tmp_path
):
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    panel.set_targets(["scaled"])
    _replay(panel, qtbot)
    graph = panel._session.recorder.graph()
    assert graph.steps() == ["scaled"]


def test_a_replay_on_another_layer_is_a_step_of_its_own(viewer, panel, qtbot, tmp_path):
    first = viewer.add_image(np.ones((8, 8)), name="image")
    second = viewer.add_image(np.full((8, 8), 0.1), name="second")
    panel.load(_recipe(tmp_path))
    panel.bind("image", first)
    _replay(panel, qtbot)
    panel.bind("image", second)
    _replay(panel, qtbot)
    names = [layer.name for layer in viewer.layers]
    assert "mask" in names and "mask_2" in names
    assert not viewer.layers["mask_2"].data.any()  # 0.1 * 3 is not above 1


def test_the_record_panel_shows_what_was_replayed(viewer, panel, qtbot, tmp_path):
    record = RecordPanel(viewer)
    viewer.window.add_dock_widget(record)
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    _replay(panel, qtbot)
    steps = [
        record._table.item(row, 1).text() for row in range(record._table.rowCount())
    ]
    assert steps == ["scaled", "mask"]


def test_a_path_input_is_asked_for(viewer, panel, qtbot, tmp_path):
    weights = tmp_path / "weights.npy"
    np.save(weights, np.array(5.0))
    recipe = _recipe(
        tmp_path,
        ("weighted", weighted, {"weights": weights}),
        targets=["weighted"],
    )
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(recipe)
    assert panel.why_not() == "Choose what stands for weighted.weights."
    other = tmp_path / "other.npy"
    np.save(other, np.array(2.0))
    panel.bind("weighted.weights", str(other))
    _replay(panel, qtbot)
    assert np.allclose(viewer.layers["weighted"].data, 1 * 3 * 2.0)


def test_a_failure_partway_keeps_what_ran(viewer, panel, qtbot, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    recipe = _recipe(tmp_path, ("broken", scale, {}), targets=["broken"])
    # Recorded with an op that worked; replayed with one that does not.
    recipe["steps"]["broken"]["op"] = OpSpec.from_op(stops).name
    panel.load(recipe)
    panel._every.setChecked(True)
    _replay(panel, qtbot)
    graph = panel._session.recorder.graph()
    assert graph.current("scaled") is not None
    assert graph.runs_of("broken")[-1].status == "failed"
    assert "RuntimeError" in panel._status.text()
    assert "scaled" in [layer.name for layer in viewer.layers]


def test_a_cancel_stops_before_the_next_step(viewer, panel, qtbot, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    recorder = Recorder(SimpleStore(tmp_path / "recorded"), InProcess())
    image = np.ones((8, 8), dtype="float32")
    recorder.root("image", image)
    first = recorder.run(interrupted, "first", {"image": image})
    recorder.run(above, "mask", {"image": first})
    panel.load(export(recorder.graph(), ["mask"], recorder.store))
    WHILE_RUNNING[:] = [panel.cancel]
    try:
        _replay(panel, qtbot)
    finally:
        WHILE_RUNNING.clear()
    graph = panel._session.recorder.graph()
    # The step running when Cancel was pressed is recorded as cancelled, and
    # the next one never starts.
    assert graph.runs_of("first")[-1].status == "cancelled"
    assert graph.runs_of("mask") == []
    assert panel._status.text() == "Cancelled after 1 steps."


def test_an_unknown_op_disables_run_and_says_why(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    recipe = _recipe(tmp_path)
    recipe["steps"]["mask"]["op"] = "nowhere:at_all"
    panel.load(recipe)
    assert "nowhere:at_all" in panel.why_not()
    assert not panel._run.isEnabled()


def test_removing_a_bound_layer_unbinds_it(viewer, panel, tmp_path):
    image = viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    viewer.layers.remove(image)
    assert panel._bindings == {}
    assert "Choose what stands for image" in panel.why_not()


def test_each_step_shows_its_recorded_parameters(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    assert set(panel._forms) == {"scaled", "mask"}
    inputs = panel._forms["scaled"][0]
    assert {w.name: w.value for w in inputs.widgets} == {"factor": 3.0}
    assert panel.overrides() == {}


def test_a_changed_parameter_is_what_runs_and_is_recorded(
    viewer, panel, qtbot, tmp_path
):
    viewer.add_image(np.full((8, 8), 0.5, dtype="float32"), name="image")
    panel.load(_recipe(tmp_path))
    panel.set_param("scaled", "factor", 1.0)
    assert panel.overrides() == {"scaled": {"factor": 1.0}}
    assert "1 changed" in panel._params.itemText(0)
    _replay(panel, qtbot)
    graph = panel._session.recorder.graph()
    assert graph.current("scaled").params == {"factor": 1.0}
    assert not viewer.layers["mask"].data.any()  # 0.5 is not above 1


def test_an_edit_survives_ticking_another_target(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    panel.set_param("scaled", "factor", 7.0)
    panel.set_targets(["scaled"])
    assert set(panel._forms) == {"scaled"}
    assert panel.overrides() == {"scaled": {"factor": 7.0}}


def test_a_parameter_left_at_its_default_is_not_an_override(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    # 'mask' recorded no level, so the op's default stands and is shown.
    inputs = panel._forms["mask"][0]
    assert {w.name: w.value for w in inputs.widgets} == {"level": 1.0}
    assert panel.overrides() == {}


def test_running_on_each_selected_layer(viewer, panel, qtbot, tmp_path):
    a = viewer.add_image(np.ones((8, 8)), name="a")
    bright = viewer.add_image(np.full((8, 8), 500.0), name="bright")
    b = viewer.add_image(np.full((8, 8), 0.1), name="b")
    recorder = Recorder(SimpleStore(tmp_path / "recorded"), InProcess())
    image = np.ones((8, 8), dtype="float32")
    recorder.root("image", image)
    scaled = recorder.run(picky, "scaled", {"image": image})
    recorder.run(above, "mask", {"image": scaled})
    panel.load(export(recorder.graph(), ["mask"], recorder.store))
    with qtbot.waitSignal(panel.finished, timeout=60_000):
        panel.run_on_each([a, bright, b])
    qtbot.wait(50)
    status = panel._status.text()
    assert status.startswith("Ran on 2 of 3 layers.")
    assert "bright: ValueError: too bright" in status
    graph = panel._session.recorder.graph()
    sources = {
        graph.sources_of(graph.current(step))["image"]
        for step in graph.steps()
        if step.startswith("scaled") and graph.current(step)
    }
    assert sources == {"root:a", "root:b"}
    names = [layer.name for layer in viewer.layers]
    assert {"mask", "mask_2"} <= set(names)


def test_running_on_each_wants_one_image_input(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    panel.load(_recipe(tmp_path))
    assert panel.why_not_each() is None
    recipe = _recipe(tmp_path)
    recipe["roots"]["second"] = {"ndim": 2, "dtype": "float32"}
    recipe["steps"]["mask"]["inputs"]["image"] = {"root": "second"}
    recipe["targets"] = ["scaled", "mask"]  # two steps, two inputs
    panel.load(recipe)
    assert "one image input" in panel.why_not_each()


def test_selected_layers_that_do_not_fit_are_left_out(viewer, panel, tmp_path):
    viewer.add_image(np.ones((8, 8)), name="image")
    stack = viewer.add_image(np.ones((3, 8, 8)), name="stack")
    panel.load(_recipe(tmp_path))
    panel.run_on_each([stack])
    assert "None of the selected layers fits" in panel._status.text()


def test_the_record_panels_last_export_can_be_opened(viewer, panel, tmp_path):
    record = RecordPanel(viewer)
    viewer.window.add_dock_widget(record)
    image = viewer.add_image(np.ones((8, 8)), name="image")
    session = panel._session
    session.recording = True
    steps: list[str] = []
    result = session.executor.run(
        scale, {"image": image.data}, on_recorded=steps.append
    )
    session.draw(steps[0], "result", result, "image", {"name": steps[0]})
    assert not panel._last.isEnabled()
    record.export_recipe(tmp_path / "exported.json")
    assert panel._last.isEnabled()
    panel._last.click()
    assert list(panel._recipe["steps"]) == [steps[0]]
