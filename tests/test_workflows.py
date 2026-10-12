"""The Workflows panel: a combo box per stage, with that stage's own inputs.

Nothing here runs a workflow. What is worth pinning is the rendering -- and
above all that a workflow whose stages both take the image asks for it once.
Every workflow in the collection is checked rather than one chosen by name,
so a new one that gets the binding wrong fails here rather than in use.
"""

from __future__ import annotations

import pytest

from skop_napari import OpsPanel, WorkflowsPanel


@pytest.fixture
def panel(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    widget = WorkflowsPanel(viewer)
    qtbot.addWidget(widget.native)
    return widget


def _labels(panel):
    return sorted(panel._by_label)


def test_the_two_panels_partition_the_collection(panel, make_napari_viewer):
    ops = OpsPanel(make_napari_viewer())
    assert set(_labels(panel)).isdisjoint(_labels(ops))
    assert all(spec.is_workflow for spec in panel._by_label.values())
    assert not any(spec.is_workflow for spec in ops._by_label.values())


def test_there_is_something_to_show(panel):
    assert _labels(panel)


def test_a_stage_is_drawn_per_chooser(panel):
    for label in _labels(panel):
        panel._picker.value = label
        choosers = [p.name for p in panel.spec.inputs if p.choices is not None]
        assert [s.name for s in panel._inputs.extra] == choosers


def test_bound_parameters_get_no_widget(panel):
    """The complication this whole shape exists for.

    Both stages of detect_then_mask take the image, and the workflow supplies
    it, so neither stage may offer a widget for it -- otherwise the panel
    shows three image combos for one image.
    """
    for label in _labels(panel):
        panel._picker.value = label
        for stage in panel._inputs.extra:
            shown = {w.name for w in stage._inputs.widgets}
            assert shown.isdisjoint(stage._binds), (
                f"{label}/{stage.name} offers a widget for a bound parameter"
            )


def test_the_image_is_asked_for_exactly_once(panel):
    for label in _labels(panel):
        panel._picker.value = label
        names = [w.name for w in panel._inputs.widgets]
        names += [w.name for s in panel._inputs.extra for w in s._inputs.widgets]
        assert names.count("image") == 1, f"{label} asks for the image {names}"


def test_the_call_names_every_parameter_the_workflow_declares(panel):
    for label in _labels(panel):
        panel._picker.value = label
        supplied = set(panel._inputs.values())
        declared = {p.name for p in panel.spec.inputs}
        assert supplied == declared, f"{label}: {declared - supplied} unfilled"


def test_changing_a_choice_redraws_that_stage(panel):
    panel._picker.value = "scikit-ops › deconvolve: deconvolve_with_psf"
    stage = panel._inputs.extra[0]
    first, second = stage._param.choices.labels[:2]

    stage._combo.value = first
    before = [w.name for w in stage._inputs.widgets]
    stage._combo.value = second
    after = [w.name for w in stage._inputs.widgets]

    # gaussian_psf and gibson_lanni share no parameter names at all, which is
    # what makes this a test of the rebuild rather than of the combo.
    assert before and after and before != after
    assert panel._inputs.values()["psf_op"] is stage._param.choices.op(second)


def test_the_chosen_op_reaches_the_call(panel):
    panel._picker.value = "scikit-ops › mask: detect_then_mask"
    detector, masker = panel._inputs.extra
    for label in detector._param.choices.labels:
        detector._combo.value = label
        assert panel._inputs.values()["detector"] is detector._param.choices.op(label)
    assert panel._inputs.values()["masker"] is masker.op


def test_the_combo_says_where_the_choice_will_run(panel):
    """Picking "gpu" for the first time means an environment build, and that
    is worth knowing before pressing Run rather than four minutes after."""
    panel._picker.value = "scikit-ops › deconvolve: deconvolve_with_psf"
    stage = panel._inputs.extra[1]
    stage._combo.value = "gpu"
    assert "cupy" in stage._combo.tooltip
    stage._combo.value = "cpu"
    assert "skimage" in stage._combo.tooltip


def test_notes_say_a_workflow_runs_on_the_host(panel):
    for label in _labels(panel):
        panel._picker.value = label
        assert "host" in panel._notes.value


def test_a_workflow_that_deconvolves_offers_tiling(panel):
    import numpy as np

    panel._viewer.add_image(np.zeros((32, 512, 512), np.float32), name="volume")
    panel.reset_choices()  # not docked, so napari doesn't do it
    panel._picker.value = next(
        label
        for label, spec in panel._by_label.items()
        if spec.function == "deconvolve_with_psf"
    )
    decon = next(s for s in panel._inputs.extra if s.name == "decon_op")
    # The workflow's own default for its deconvolver: tiles need it.
    assert decon.values()["noncirc"] is True
    # The PSF does not exist until the workflow runs, so the plan shown is
    # estimated with 10; the run leaves the overlap to the op.
    assert not panel._memory_box._explicitly_hidden
    assert panel._tile_plan.overlap == 10
    assert panel._tiler()[0].overlap is None
    panel._memory.value = "200M"
    assert panel._tile_plan.calls > 1
    assert "estimated" in panel._tiles.value
