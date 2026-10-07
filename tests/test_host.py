"""The recording host, without a viewer.

What the host decides -- what a root is, what gets drawn, what counts as a hand
edit -- is asserted here against a display that is a list. Only the facts that
need napari are asserted with a real viewer.
"""

from __future__ import annotations

import numpy as np
import pytest

from skop import OpSpec, Role, op
from skop.types import ImageOf, LabelsOf
from skop_core import InProcess, Recorder, SimpleStore
from skop_napari._host import SKOP, Host, UnknownLayer, UnknownOp


@op()
def blur(image: ImageOf[np.ndarray], sigma: float = 1.0) -> ImageOf[np.ndarray]:
    """Add sigma, so a result is easy to recognise."""
    return np.asarray(image, dtype=float) + sigma


@op()
def threshold(image: ImageOf[np.ndarray]) -> LabelsOf[np.ndarray]:
    """Anything above zero, as 0/1 labels."""
    return (np.asarray(image) > 0).astype(np.uint8)


BLUR = OpSpec.from_op(blur).name
THRESHOLD = OpSpec.from_op(threshold).name
REGISTRY = {BLUR: blur, THRESHOLD: threshold}


class FakeLayer:
    """A layer as the host sees one: a name, data and a metadata dict."""

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.metadata = {}


class FakeDisplay:
    """A viewer that is a list. The second adapter at the host's seam."""

    def __init__(self, *layers):
        self._layers = list(layers)
        self.roles = {}

    def layers(self):
        return list(self._layers)

    def show(self, name, data, *, step, role=None, metadata=None):
        self.roles[step] = role
        for layer in self._layers:
            if (layer.metadata.get(SKOP) or {}).get("step") == step:
                layer.data = data
                layer.metadata.update(metadata or {})
                return layer
        layer = FakeLayer(name, data)
        layer.metadata.update(metadata or {})
        self._layers.append(layer)
        return layer

    def by_name(self, name):
        return next(layer for layer in self._layers if layer.name == name)

    def names(self):
        return [layer.name for layer in self._layers]


def an_image():
    return np.arange(16.0).reshape(4, 4)


def make_host(tmp_path, *layers):
    display = FakeDisplay(*layers)
    recorder = Recorder(SimpleStore(tmp_path / "store"), InProcess())
    return Host(recorder, display, REGISTRY)


def test_a_new_layer_becomes_a_root(tmp_path):
    image = an_image()
    host = make_host(tmp_path, FakeLayer("image", image))
    assert host.declare_roots() == ["image"]
    handle = host.recorder.graph().current_root("image").handle
    assert host.recorder.store.handle_of(image) == handle
    assert host.recorder.store.get(handle) is image  # held, not copied
    link = host.display.by_name("image").metadata[SKOP]
    assert link["step"] is None
    assert link["handle"] == handle
    assert link["id"]


def test_declaring_again_changes_nothing(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    assert host.declare_roots() == []
    assert len(host.recorder.graph().roots_of("image")) == 1


def test_a_root_is_known_by_its_id_not_its_name(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    root = host.display.by_name("image")
    root.name = "the original"
    assert host.array_for(root) is root.data


def test_a_layer_the_host_never_declared_is_refused(tmp_path):
    host = make_host(tmp_path)
    with pytest.raises(UnknownLayer, match="declare_roots"):
        host.run("blurred", BLUR, {"image": FakeLayer("stray", an_image())})


def test_a_step_the_host_cannot_name_is_refused(tmp_path):
    host = make_host(tmp_path)
    with pytest.raises(UnknownOp):
        host.run("blurred", "not:an-op", {})


def test_running_draws_each_step_with_the_role_its_op_declares(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    host.run("labels", THRESHOLD, {"image": host.display.by_name("blurred")})
    assert host.display.names() == ["image", "blurred", "labels"]
    assert host.display.roles["blurred"] is Role.image
    assert host.display.roles["labels"] is Role.labels
    graph = host.recorder.graph()
    assert graph.sources_of(graph.current("labels")) == {"image": "blurred"}


def test_showing_again_moves_the_layer_rather_than_adding_one(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    for sigma in (1.0, 2.0, 3.0):
        host.run(
            "blurred",
            BLUR,
            {"image": host.display.by_name("image"), "sigma": sigma},
        )
    assert host.display.names() == ["image", "blurred"]
    assert len(host.recorder.graph().runs_of("blurred")) == 3


def test_renaming_the_layer_changes_nothing(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    host.display.by_name("blurred").name = "my smoothing"
    host.run("blurred", BLUR, {"image": host.display.by_name("image"), "sigma": 4.0})
    assert host.display.names() == ["image", "my smoothing"]


def test_the_hosts_own_write_is_not_a_hand_edit(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    assert host.data_changed(host.display.by_name("blurred")) is False
    assert not host.recorder.graph().is_edited("blurred")


def test_a_hand_edit_marks_the_step_and_affects_its_consumer(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    host.run("labels", THRESHOLD, {"image": host.display.by_name("blurred")})
    painted = host.display.by_name("blurred")
    painted.data = painted.data + 1.0
    assert host.data_changed(painted) is True
    assert host.recorder.graph().is_edited("blurred")
    assert host.recorder.graph().is_affected("labels")


def test_reverting_puts_the_recorded_output_back(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    layer = host.display.by_name("blurred")
    recorded = layer.data.copy()
    layer.data = layer.data + 99.0
    assert host.data_changed(layer) is True
    host.revert("blurred")
    assert not host.recorder.graph().is_edited("blurred")
    assert np.array_equal(host.display.by_name("blurred").data, recorded)


def test_a_root_that_moves_stales_its_consumer(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    root = host.display.by_name("image")
    root.data = root.data + 1.0
    assert host.data_changed(root) is True
    assert host.recorder.graph().is_stale("blurred")
    assert host.produce(["blurred"]) == ["blurred"]
    assert not host.recorder.graph().is_stale("blurred")


def test_a_rerun_adds_a_run_rather_than_moving_the_step(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    host.rerun("blurred")
    assert len(host.recorder.graph().runs_of("blurred")) == 2


def test_the_view_reports_the_steps_and_their_badges(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    drawn = host.view()
    assert [node.step for node in drawn.nodes] == ["blurred"]
    assert [node.badge for node in drawn.nodes] == ["fresh"]


def test_a_rewrapped_layer_resolves_to_what_is_stored(tmp_path):
    """A lazy layer hands out a fresh array each read; its link is the answer."""
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    layer = host.display.by_name("blurred")
    stored = host.recorder.store.get(layer.metadata[SKOP]["handle"])
    layer.data = np.array(layer.data)  # a new object, as a lazy layer gives
    assert np.array_equal(host.array_for(layer), stored)


def test_an_edited_layer_is_refused_rather_than_quietly_replaced(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    layer = host.display.by_name("blurred")
    layer.data = layer.data + 1.0
    host.data_changed(layer)
    assert host.recorder.graph().is_edited("blurred")
    with pytest.raises(UnknownLayer, match="Accept the edit"):
        host.array_for(layer)


def test_keeping_an_edit_lets_the_layer_be_run_on(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    host.run("labels", THRESHOLD, {"image": host.display.by_name("blurred")})
    layer = host.display.by_name("blurred")
    layer.data = layer.data * 0 - 1.0  # nothing above zero any more
    host.data_changed(layer)
    with pytest.raises(UnknownLayer):
        host.array_for(layer)

    assert host.keep("blurred") == ["result"]
    graph = host.recorder.graph()
    assert graph.is_accepted("blurred") and not graph.is_edited("blurred")
    assert graph.is_stale("labels")
    assert host.array_for(layer) is layer.data
    assert host.produce(["labels"]) == ["labels"]
    labels = host.recorder.store.get(
        host.recorder.graph().current("labels").outputs["result"]
    )
    assert not labels.any()


def test_keeping_a_painted_layer_keeps_the_strokes(tmp_path):
    # Painting changes the array in place: the object is the one recorded,
    # its pixels are not.
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("labels", THRESHOLD, {"image": host.display.by_name("image")})
    layer = host.display.by_name("labels")
    layer.data[0, 0] = 7
    assert host.painted(layer) is True
    host.keep("labels")
    kept = host.recorder.store.get(host.recorder.graph().accepted("labels"))
    assert kept[0, 0] == 7


def test_reverting_a_kept_edit_draws_what_the_run_made(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    layer = host.display.by_name("blurred")
    recorded = layer.data.copy()
    layer.data = layer.data + 99.0
    host.data_changed(layer)
    host.keep("blurred")
    host.revert("blurred")
    graph = host.recorder.graph()
    assert not graph.is_accepted("blurred") and not graph.is_edited("blurred")
    assert np.array_equal(host.display.by_name("blurred").data, recorded)


def test_a_kept_step_is_drawn_from_its_kept_pixels(tmp_path):
    host = make_host(tmp_path, FakeLayer("image", an_image()))
    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    layer = host.display.by_name("blurred")
    layer.data = np.full_like(layer.data, 5.0)
    host.data_changed(layer)
    host.keep("blurred")
    layer.data = np.zeros_like(layer.data)  # the layer is cleared by mistake
    host.show("blurred")
    assert (host.display.by_name("blurred").data == 5.0).all()


class Lazy:
    """Array-shaped without being numpy, as a dask or zarr layer's data is."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape, self.dtype = self._data.shape, self._data.dtype
        self.ndim, self.size = self._data.ndim, self._data.size
        self.reads = 0

    def __getitem__(self, key):
        return self._data[key]

    def __array__(self, dtype=None, copy=None):
        self.reads += 1
        return self._data


def test_a_lazy_root_is_held_as_itself_and_never_read(tmp_path):
    lazy = Lazy(an_image())
    host = make_host(tmp_path, FakeLayer("image", lazy))
    host.declare_roots()
    root = host.recorder.graph().current_root("image")
    assert host.recorder.store.get(root.handle) is lazy
    assert host.array_for(host.display.by_name("image")) is lazy
    assert lazy.reads == 0


# -- replaying a recipe onto layers ------------------------------------------


def _recorded_recipe(host):
    from skop_core import export

    host.declare_roots()
    host.run("blurred", BLUR, {"image": host.display.by_name("image")})
    host.run("labels", THRESHOLD, {"image": host.display.by_name("blurred")})
    graph = host.recorder.graph()
    return export(graph, ["labels"], host.recorder.store)


def test_a_recipe_replays_onto_another_layer(tmp_path):
    from skop_core import prepare

    other = FakeLayer("other", an_image() - 8.0)
    host = make_host(tmp_path, FakeLayer("image", an_image()), other)
    recipe = _recorded_recipe(host)
    prepared = prepare(recipe, REGISTRY)
    done = host.replay(prepared, {"image": other})

    assert done.steps == {"blurred": "blurred_2", "labels": "labels_2"}
    graph = host.recorder.graph()
    assert graph.sources_of(graph.current("blurred_2")) == {"image": "root:other"}
    # Only the target is drawn, as the layer standing for its session step.
    assert host.display.names()[-1] == "labels_2"
    assert "blurred_2" not in host.display.names()


def test_every_step_can_be_drawn(tmp_path):
    from skop_core import prepare

    other = FakeLayer("other", an_image())
    host = make_host(tmp_path, FakeLayer("image", an_image()), other)
    prepared = prepare(_recorded_recipe(host), REGISTRY)
    host.replay(prepared, {"image": other}, every_step=True)
    assert {"blurred_2", "labels_2"} <= set(host.display.names())


def test_replaying_onto_the_same_layer_adds_runs_not_steps(tmp_path):
    from skop_core import prepare

    host = make_host(tmp_path, FakeLayer("image", an_image()))
    prepared = prepare(_recorded_recipe(host), REGISTRY)
    done = host.replay(prepared, {"image": host.display.by_name("image")})
    assert done.steps == {"blurred": "blurred", "labels": "labels"}
    assert len(host.recorder.graph().runs_of("labels")) == 2


def test_a_layer_never_declared_is_declared_when_bound(tmp_path):
    from skop_core import prepare

    host = make_host(tmp_path, FakeLayer("image", an_image()))
    prepared = prepare(_recorded_recipe(host), REGISTRY)
    late = FakeLayer("late", an_image())
    host.display._layers.append(late)
    host.replay(prepared, {"image": late})
    assert host.recorder.graph().current_root("late") is not None


def test_an_edited_layer_is_refused_as_an_input(tmp_path):
    from skop_core import prepare

    host = make_host(tmp_path, FakeLayer("image", an_image()))
    prepared = prepare(_recorded_recipe(host), REGISTRY)
    blurred = host.display.by_name("blurred")
    blurred.data = blurred.data + 1.0
    host.data_changed(blurred)
    with pytest.raises(UnknownLayer, match="Accept the edit"):
        host.replay(prepared, {"image": blurred})


def test_a_replay_can_chain_onto_a_steps_layer(tmp_path):
    from skop_core import prepare

    host = make_host(tmp_path, FakeLayer("image", an_image()))
    prepared = prepare(_recorded_recipe(host), REGISTRY)
    done = host.replay(prepared, {"image": host.display.by_name("blurred")})
    graph = host.recorder.graph()
    assert graph.sources_of(graph.current(done.steps["blurred"])) == {
        "image": "blurred"
    }
