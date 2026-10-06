"""The Ops panel, against a real napari viewer.

These tests run ops for real, in their own Appose environments, so they need
the 'minimal' environment built. That is the cheapest one skop has -- python
plus numpy plus appose -- and building it once makes the difference between
testing the plumbing and testing a mock of it.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from skop_napari import OpsPanel


@contextlib.contextmanager
def errors_raised():
    """Collect the error notifications raised inside the block.

    NB: notification_manager.records accumulates across the whole session,
    so a test that asserts on its length has to slice off what was already
    there or it will see the previous test's errors.
    """
    from napari.utils.notifications import notification_manager

    collected: list = []
    with notification_manager:
        already = len(notification_manager.records)
        yield collected
        collected.extend(
            n for n in notification_manager.records[already:] if n.severity == "error"
        )


@pytest.fixture
def panel(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = OpsPanel(viewer)
    viewer.window.add_dock_widget(widget)
    yield widget
    # Each panel holds its own Runner, and so its own worker processes.
    widget._runner.close()


def _choose(panel, function_name):
    label = next(
        lbl for lbl, spec in panel._by_label.items() if spec.function == function_name
    )
    panel._picker.value = label


def test_napari_supplies_the_viewer_when_it_builds_the_widget(make_napari_viewer):
    # Regression: the panel used to take `viewer: Any = None`, which napari
    # does not recognise, so it was built with no viewer and every layer
    # output was silently discarded. Built here through napari's own logic
    # rather than by calling OpsPanel directly, which is what hid the bug.
    from napari._qt.qt_main_window import _instantiate_dock_widget

    viewer = make_napari_viewer()
    widget = _instantiate_dock_widget(OpsPanel, viewer)
    try:
        assert widget._viewer is not None
    finally:
        widget._runner.close()


def test_layer_outputs_with_no_viewer_are_reported_not_dropped(qtbot):
    # A panel with no viewer must not look like a panel whose op did nothing.
    widget = OpsPanel()
    try:
        _choose(widget, "find_nothing")
        with errors_raised() as errors:
            widget._on_done((np.zeros((4, 4), np.uint16), np.zeros((0, 3), np.int32)))
    finally:
        widget._runner.close()

    assert len(errors) == 1
    assert "labels" in str(errors[0].message)
    assert "napari_viewer" in str(errors[0].message)


def test_lists_the_collection(panel):
    functions = {spec.function for spec in panel._by_label.values()}
    assert {"otsu", "stardist2d_fluo", "unseg", "add"} <= functions


def test_choosing_an_op_rebuilds_the_inputs(panel):
    _choose(panel, "otsu")
    assert [w.name for w in panel._inputs.widgets] == [
        "image",
        "invert",
        "label_objects",
    ]

    _choose(panel, "add")
    assert [w.name for w in panel._inputs.widgets] == ["a", "b"]
    assert "Environment: minimal" in panel._notes.value


def test_image_inputs_offer_the_viewers_layers(panel, make_napari_viewer):
    viewer = panel._viewer
    viewer.add_image(np.zeros((16, 16), dtype=np.uint8), name="first")
    viewer.add_image(np.ones((16, 16), dtype=np.uint8), name="second")

    _choose(panel, "otsu")
    image = next(w for w in panel._inputs.widgets if w.name == "image")
    # The role annotation is what turned this into a layer combo box. It
    # offers the layers' data, not the layers, which is what the op wants.
    assert len(image.choices) == 2
    assert [float(choice.mean()) for choice in image.choices] == [0.0, 1.0]


def test_a_layer_added_later_shows_up_in_the_combo(panel):
    _choose(panel, "otsu")
    image = next(w for w in panel._inputs.widgets if w.name == "image")
    assert image.choices == ()

    # napari refreshes the choices of the widgets it docks, so an op panel
    # built before the user opened an image still offers it.
    panel._viewer.add_image(np.zeros((4, 4), dtype=np.uint8), name="later")
    assert len(image.choices) == 1


def test_running_an_op_adds_its_output_as_the_right_layer_type(panel, qtbot):
    viewer = panel._viewer
    viewer.add_image(np.zeros((8, 8), dtype=np.uint16), name="blank")
    _choose(panel, "find_nothing")
    # The sole image layer is already selected: a combo built with choices
    # present picks the first one, which is what a user would have to do.
    image = next(w for w in panel._inputs.widgets if w.name == "image")
    assert image.value is viewer.layers["blank"].data

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()

    kinds = {layer.name: type(layer).__name__ for layer in viewer.layers}
    # labels -> Labels and points -> Points, from the op's role annotations.
    assert kinds["labels [find_nothing]"] == "Labels"
    assert kinds["points [find_nothing]"] == "Points"


def test_scalar_outputs_go_to_the_results_panel(panel, qtbot):
    _choose(panel, "add")
    next(w for w in panel._inputs.widgets if w.name == "a").value = 17
    next(w for w in panel._inputs.widgets if w.name == "b").value = 25

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()

    # NB: assert on contents, not on .visible -- a magicgui widget reports
    # the Qt state, and the test viewer is never actually shown.
    assert [(w.name, w.value) for w in panel._results] == [("result", "42")]
    # A scalar has no layer to be, so nothing was added to the viewer.
    assert len(panel._viewer.layers) == 0


def test_progress_reaches_the_panel_and_cancel_stops_the_op(panel, qtbot):
    panel._viewer.add_image(np.ones((10, 10), dtype=np.float32), name="ones")
    _choose(panel, "slow_sum")
    next(w for w in panel._inputs.widgets if w.name == "steps").value = 60

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()
        # Progress is reported from the worker process, relayed by an Appose
        # listener thread, and must arrive on the GUI thread to be shown.
        qtbot.waitUntil(
            lambda: "Summing chunk" in (panel._progress.label or ""),
            timeout=120_000,
        )
        assert panel._run is not None
        panel._stop()

    # slow_sum polls cancel_requested() and returns its partial sum, so a
    # canceled op still delivers -- 100.0 is what a full run would total.
    total = float(panel._results[0].value)
    assert 0.0 < total < 100.0


def test_a_failing_op_reports_through_napari(panel, qtbot):
    # No image layer exists, so scale gets None and fails in the worker.
    _choose(panel, "scale")
    assert next(w for w in panel._inputs.widgets if w.name == "image").value is None

    with (
        errors_raised() as errors,
        qtbot.waitSignal(panel.finished, timeout=300_000),
    ):
        panel._start()

    assert len(errors) == 1
    # The op failed in another process, and the notification carries that
    # process's traceback -- which is the whole reason not to use a LineEdit.
    message = str(errors[0].message)
    assert "Task failed" in message
    assert "toy.py" in message and "image * factor" in message
    assert "unsupported operand type(s)" in message

    # Nothing was written into the panel, and it is usable again.
    assert len(panel._results) == 0
    assert panel._button.enabled


def test_build_progress_drives_the_progress_bar(panel):
    # PixiInstallMonitor reports determinate progress per phase. These are
    # the real titles it emits, in order.
    panel._on_build_progress("Installing conda packages", 0, 30)
    assert panel._progress.label == "Installing conda packages"
    assert panel._progress.max == 30

    panel._on_build_progress("Installing conda packages", 21, 30)
    assert panel._progress.value == 21


def test_build_log_noise_is_logged_but_not_shown(panel, caplog):
    # Appose runs pixi under -vv to drive the monitor, so most of the stderr
    # stream is pixi's debug log rather than status worth showing.
    with caplog.at_level("INFO", logger="skop_napari"):
        panel._on_build_text(
            "DEBUG pixi_config: Loading config from /etc/pixi/config.toml\n"
            " INFO pixi_core::lock_file::update: Installed environment\n"
        )
    assert panel._progress.label == ""
    assert "pixi_config" in caplog.text  # Still recoverable from the log.

    # A human-facing line does reach the bar.
    panel._on_build_text("✔ The default environment has been installed.\n")
    assert panel._progress.label == "✔ The default environment has been installed."


def test_build_callbacks_are_registered_on_the_runner(panel):
    assert panel._runner._build_progress
    assert panel._runner._build_output
    assert panel._runner._build_error


def test_build_text_crosses_from_the_build_thread_to_the_gui(panel, qtbot):
    import threading

    # Environments are built inside run(), on the worker thread, so every
    # build callback arrives off the GUI thread and has to be marshalled.
    subscriber = panel._runner._build_error[0]
    threading.Thread(target=subscriber, args=("✔ installed\n",)).start()

    qtbot.waitUntil(lambda: panel._progress.label == "✔ installed", timeout=10_000)


# -- axis awareness --------------------------------------------------------


def _row(panel, param="image"):
    return next(r for r in panel._adapt._rows if r.param.name == param)


def test_only_ops_that_declared_axes_get_an_adaptation_row(panel):
    _choose(panel, "add")
    assert panel._adapt._rows == []
    _choose(panel, "quadrants")
    assert [r.param.name for r in panel._adapt._rows] == ["image"]


def test_a_stack_maps_its_inner_axes_and_offers_the_rest(panel):
    panel._viewer.add_image(np.zeros((3, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")

    row = _row(panel)
    # Nobody named these axes, so nobody pretends to have: they keep the names
    # napari gives them, and nothing is written back onto the layer.
    assert row._axes.value == "-3 -2 -1"
    assert "skop_axes" not in panel._viewer.layers["stack"].metadata

    # y and x filled from the plane the viewer is showing; the slider axis is
    # left over, with a control of its own.
    assert row.plan.mapping == (1, 2)
    assert [c.value for c in row._slots] == [1, 2]
    assert [c.label for c in row._extra] == ["-3"]
    # The default keeps everything, so nothing needs confirming.
    assert row.plan.uses_all_data
    assert row.plan.iterate == (0,)
    # An unnamed axis makes no claim, so feeding it to y is nothing to warn of.
    assert row.warnings == ()


def test_the_displayed_plane_fills_the_spatial_slots(panel):
    # The heuristic's whole remaining job: whatever the user has rolled to the
    # front is what the op gets, names or no names.
    panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    assert _row(panel).plan.mapping == (1, 2)

    panel._viewer.dims.order = (1, 2, 0)  # axes 2 and 0 on screen, 1 sliding
    panel._replan()

    row = _row(panel)
    assert row.plan.mapping == (2, 0)
    assert row.plan.iterate == (1,)


def test_the_displayed_plane_outranks_the_axis_names(panel):
    # Somebody who has rolled the dims round to look at the zx plane means to
    # run on the zx plane. skop says so in a warning rather than refusing.
    layer = panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32))
    layer.axis_labels = ("z", "y", "x")
    _choose(panel, "quadrants")
    assert _row(panel).plan.mapping == (1, 2)

    panel._viewer.dims.order = (1, 0, 2)  # z and x displayed, y on the slider
    panel._replan()

    row = _row(panel)
    assert row.plan.mapping == (0, 2)
    assert row.warnings == ("y is being fed the z axis",)


def test_a_3d_view_is_read_as_well(panel):
    # Rung four only read the viewer in 2-D view, because it was after a
    # plane. A 2-D op in a 3-D view takes the two innermost axes on screen.
    panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    panel._viewer.dims.ndisplay = 3
    panel._viewer.dims.order = (1, 0, 2)
    panel._replan()

    assert _row(panel).plan.mapping == (0, 2)


def test_a_leftover_axis_can_be_switched_to_the_current_position(panel):
    panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    panel._viewer.dims.set_current_step(0, 3)
    panel._replan()

    row = _row(panel)
    row._extra[0].value = "select"

    assert row.plan.select == ((0, 3),)
    assert "axis 0=3" in row.plan.summary
    assert not row.plan.uses_all_data


def test_moving_the_slider_moves_the_selected_position(panel):
    # Regression: the plan was made once and never revisited, so "current
    # position" meant wherever the viewer happened to be when the axis was
    # switched, and Run used that stale slice.
    panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    row = _row(panel)
    row._extra[0].value = "select"

    panel._viewer.dims.set_current_step(0, 3)
    assert row.plan.select == ((0, 3),)

    panel._viewer.dims.set_current_step(0, 1)
    assert row.plan.select == ((0, 1),)
    assert "axis 0=1" in row.plan.summary


def test_moving_the_slider_leaves_a_position_free_plan_alone(panel):
    # The default plan iterates, so it says the same thing wherever the
    # sliders are; re-planning per slider step would be churn for nothing.
    panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    before = _row(panel).plan

    panel._viewer.dims.set_current_step(0, 3)
    assert _row(panel).plan is before


def test_remapping_the_slots_processes_cross_sections(panel):
    # The point of the mapping controls: feed the op ZY planes instead of YX
    # ones, and iterate over x. One combo per slot, set independently.
    panel._viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    row = _row(panel)
    row._axes.value = "z y x"  # named by hand, so the vocabulary is the user's

    row._slots[0].value = 0  # y <- z; the old y axis falls out to leftover
    assert row.plan.mapping == (0, 2)
    assert row.plan.iterate == (1,)

    row._slots[1].value = 1  # x <- y; now x is the leftover one
    assert row.plan.mapping == (0, 1)
    assert row.plan.iterate == (2,)
    assert row.plan.output_axes == ("x", "z", "y")
    assert row.warnings == (
        "y is being fed the z axis",
        "x is being fed the y axis",
    )


def test_remapping_onto_a_taken_axis_swaps_the_two(panel):
    # Choosing an axis another slot already holds has to displace it; swapping
    # is what a user means by dragging one onto the other.
    panel._viewer.add_image(np.zeros((8, 6), dtype=np.float32), name="plane")
    _choose(panel, "quadrants")
    row = _row(panel)

    row._slots[0].value = 1  # y <- x, which slot x was holding

    assert row.plan.mapping == (1, 0)
    assert row.plan.iterate == ()


def test_editing_the_axes_replans(panel):
    panel._viewer.add_image(np.zeros((3, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    row = _row(panel)

    # A lifetime axis is not z, and skop has never heard of it -- which is
    # fine, because an op that iterates does not need to know what it is.
    row._axes.value = "lifetime y x"
    assert row.plan.iterate == (0,)
    assert row.plan.calls == 3


def test_a_misaligned_mapping_warns_without_blocking_the_run(panel):
    # What used to be refused outright. The op wanted y and x and is getting
    # z and x; that is the user's call, so it runs and the panel says so.
    panel._viewer.add_image(np.zeros((3, 8), dtype=np.float32), name="odd")
    _choose(panel, "quadrants")
    _row(panel)._axes.value = "z x"

    assert panel._button.enabled
    assert "Check: image: y is being fed the z axis" in panel._notes.value


def test_labels_that_do_not_describe_the_array_block_the_run(panel):
    # Naming fewer axes than the array has is one of the few things left that
    # genuinely cannot be adapted, so it still disables Run and says why.
    panel._viewer.add_image(np.zeros((3, 8), dtype=np.float32), name="plane")
    _choose(panel, "quadrants")
    _row(panel)._axes.value = "x"

    assert not panel._button.enabled
    assert "1 axis label(s)" in panel._notes.value


def test_a_2d_op_runs_over_a_stack_and_labels_the_result(panel, qtbot):
    viewer = panel._viewer
    viewer.add_image(np.zeros((3, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()

    result = viewer.layers["result [quadrants]"]
    assert type(result).__name__ == "Labels"
    # One call per plane, stacked back up, with label IDs made unique.
    assert result.data.shape == (3, 8, 6)
    assert result.data.max() == 12
    # Nobody named the input's axes, so nothing is stamped on the output: a
    # made-up name would come back on the next run as a declaration.
    assert "skop_axes" not in result.metadata


def test_named_axes_are_stamped_onto_the_result(panel, qtbot):
    viewer = panel._viewer
    layer = viewer.add_image(np.zeros((3, 8, 6), dtype=np.float32), name="stack")
    layer.axis_labels = ("z", "y", "x")
    _choose(panel, "quadrants")

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()

    # Stamped, so the next op run against this layer does not guess.
    result = viewer.layers["result [quadrants]"]
    assert result.metadata["skop_axes"] == ("z", "y", "x")
    assert tuple(result.axis_labels) == ("z", "y", "x")


def test_choosing_the_current_position_runs_once(panel, qtbot):
    viewer = panel._viewer
    viewer.add_image(np.zeros((3, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    _row(panel)._extra[0].value = "select"

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()

    assert viewer.layers["result [quadrants]"].data.shape == (8, 6)


def test_a_remapped_run_processes_cross_sections(panel, qtbot):
    viewer = panel._viewer
    viewer.add_image(np.zeros((5, 8, 6), dtype=np.float32), name="stack")
    _choose(panel, "quadrants")
    row = _row(panel)
    row._axes.value = "z y x"
    row._slots[0].value = 0  # y <- z
    row._slots[1].value = 1  # x <- y, so the op sees ZY planes and x is spare

    with qtbot.waitSignal(panel.finished, timeout=300_000):
        panel._start()

    result = viewer.layers["result [quadrants]"]
    assert result.data.shape == (6, 5, 8)
    # Stamped in the caller's own vocabulary, remapping and all.
    assert tuple(result.axis_labels) == ("x", "z", "y")


def test_boxes_are_reshaped_but_never_axis_labelled(panel):
    """The seam between the two jobs the panel does to an output.

    ``layer_args_for`` adapts data to the layout napari wants; ``_layer_args``
    names the layer and stamps its axes. They meet on the box detectors'
    output, and the stamp must not land on it: a Shapes layer's dimensions are
    coordinates, not image axes, so labelling them 'y' and 'x' would be a lie
    the next op run would then read back.

    Guarded because the two were reconciled in a merge, and the thing that
    makes it work is indirect -- reshaping (N, 4) to (N, 2, 2) is what stops
    the axis count matching.
    """
    from skop import OutputSpec, Role
    from skop_napari._roles import layer_args_for

    boxes = np.array([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 9.0, 8.0]], dtype=np.float32)
    output = OutputSpec(name="boxes", type=np.ndarray, role=Role.boxes)

    data, extra = layer_args_for(output, boxes)
    assert data.shape == (2, 2, 2)
    assert extra["shape_type"] == "rectangle"

    # Two axes, which is exactly what the raw (N, 4) array's ndim would have
    # matched -- so this is the case that would mislabel if the adapted data
    # were not what gets asked about.
    panel._output_axes = ("y", "x")
    args = panel._layer_args("boxes [detector]", data)
    args.update(extra)

    assert args["name"] == "boxes [detector]"
    assert "axis_labels" not in args
    assert "metadata" not in args


def test_shapes_layer_data_reaches_an_op_as_canonical_boxes(panel):
    """The whole input side of the boxes contract, through a real layer.

    A Shapes layer holds one (V, 2) vertex array per shape; skop's ops want
    (N, 4) rows of [min_y, min_x, max_y, max_x]. Without the conversion the op
    fails inside the worker, after a model load, with a shape error.
    """
    viewer = panel._viewer
    viewer.add_image(np.zeros((64, 64), dtype=np.uint8), name="img")
    viewer.add_shapes(
        [
            np.array([[10.0, 20.0], [10.0, 40.0], [30.0, 40.0], [30.0, 20.0]]),
            np.array([[0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0]]),
        ],
        shape_type="rectangle",
        name="prompts",
    )

    _choose(panel, "mobilesam_masks")
    boxes = panel._inputs.values()["boxes"]

    assert boxes.shape == (2, 4)
    assert boxes.tolist() == [[10, 20, 30, 40], [0, 0, 5, 5]]


def test_the_mask_view_chooser_appears_only_for_ops_that_return_masks(panel):
    # NB: asserted on magicgui's intent flag, not on .visible. The panel is
    # docked into a viewer window that tests never show, so Qt reports every
    # widget in it as invisible no matter what the panel asked for.
    _choose(panel, "otsu")
    assert panel._mask_view._explicitly_hidden

    _choose(panel, "mobilesam_masks")
    assert not panel._mask_view._explicitly_hidden


def test_mask_outputs_are_projected_the_way_the_chooser_says(panel):
    from skop import OutputSpec, Role
    from skop_napari._roles import layer_args_for, layer_type_for

    # A big mask with a small one drawn inside it: the case the projection
    # strategies actually differ on.
    stack = np.zeros((2, 10, 10), dtype=np.uint8)
    stack[0, 0:8, 0:8] = 1
    stack[1, 2:5, 2:5] = 1
    output = OutputSpec(name="masks", type=np.ndarray, role=Role.masks)

    # Whichever projection is chosen, it is a Labels layer at the end.
    assert layer_type_for(output) == "labels"

    nested, _ = layer_args_for(output, stack, "2D labels (nested objects on top)")
    assert nested.shape == (10, 10)
    # The small mask is drawn over the big one, so both objects appear.
    assert nested[3, 3] == 2
    assert set(np.unique(nested)) == {0, 1, 2}

    largest, _ = layer_args_for(output, stack, "2D labels (largest object on top)")
    assert largest[3, 3] == 1
    # And the nested object is gone entirely -- the cost of this strategy,
    # asserted so nobody has to rediscover it.
    assert set(np.unique(largest)) == {0, 1}

    stacked, _ = layer_args_for(output, stack, "3D stack (nothing lost)")
    # Nothing lost: the contested pixel is on both planes, with both labels.
    assert stacked.shape == (2, 10, 10)
    assert stacked[0][3, 3] == 1
    assert stacked[1][3, 3] == 2


def test_the_3d_stack_gets_an_adjustable_z_spacing(panel):
    """The stack's first axis is an object index, so its spacing is a choice.

    At 1 a handful of masks over a wide image is a pancake and rotating it
    shows nothing, which is what the default of 10 is for. y and x stay at 1,
    matching the image the masks were found in.
    """
    from skop import OutputSpec, Role
    from skop_napari._roles import layer_args_for

    stack = np.zeros((3, 10, 10), dtype=np.uint8)
    stack[0, 0:8, 0:8] = 1
    stack[1, 2:5, 2:5] = 1
    stack[2, 6:9, 6:9] = 1
    output = OutputSpec(name="masks", type=np.ndarray, role=Role.masks)

    _, extra = layer_args_for(output, stack, "3D stack (nothing lost)")
    assert extra["scale"] == (10.0, 1.0, 1.0)

    _, extra = layer_args_for(output, stack, "3D stack (nothing lost)", 2.5)
    assert extra["scale"] == (2.5, 1.0, 1.0)

    # A 2-D projection has no third axis to space out.
    _, extra = layer_args_for(output, stack, "2D labels (nested objects on top)")
    assert "scale" not in extra


def test_the_spacing_spinner_follows_the_view_chooser(panel):
    # NB: on magicgui's intent flag rather than .visible -- see the chooser
    # test above for why.
    _choose(panel, "otsu")
    assert panel._z_spacing._explicitly_hidden

    _choose(panel, "mobilesam_masks")
    # Defaults to a 2-D projection, which has no spacing to set.
    assert panel._z_spacing._explicitly_hidden

    panel._mask_view.value = "3D stack (nothing lost)"
    assert not panel._z_spacing._explicitly_hidden
    assert panel._z_spacing.value == 10.0

    panel._mask_view.value = "2D labels (largest object on top)"
    assert panel._z_spacing._explicitly_hidden
