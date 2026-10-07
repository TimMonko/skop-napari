"""One recording session per viewer, shared by every panel that runs ops.

The recorder is not something a panel runs ops *with*. It stands where the
runner stands: the panels that run ops are pointed at ``Session.executor``,
which is the scikit-ops runner while nothing is being recorded and a runner
that writes each call down while something is. Either way the op runs the way
it always did -- in its own environment, off the GUI thread, with progress and
Cancel -- and the Record panel is only a view of what was written.

A session holds the viewer's one scikit-ops ``Runner``, so the Ops, Workflows
and Record panels share its warm workers and its environment builds.

``Session`` and ``ViewerDisplay`` need a viewer; everything they decide is in
``_host``, which does not.
"""

from __future__ import annotations

import logging
import os
import tempfile
import weakref
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import napari.viewer
from napari.layers import Layer
from psygnal import Signal
from qtpy.QtCore import QTimer

from skop import OpSpec, Role, Runner, discover
from skop_core import Graph, Recorder, RecordingRunner, SimpleStore

from ._host import SKOP, Host
from ._roles import layer_type_for_role
from ._run import resolve

_log = logging.getLogger("skop_napari")


def step_name(spec: OpSpec, graph: Graph) -> str:
    """What to record a call as: the op's function name.

    The same op can be run twice under different names one day, which needs a
    widget to type in. There is none yet, and a name derived from the op means
    running the same op again adds a run to the same step -- which is exactly
    what 'do that again' should mean.

    A function name already taken by a *different* op gets its namespace put
    back in front, so the two steps stay two.
    """
    runs = graph.runs_of(spec.function)
    if not runs or runs[-1].op == spec.name:
        return spec.function
    module, _, function = spec.name.rpartition(":")
    return f"{module.rsplit('.', 1)[-1]}_{function}"


class OpRegistry(Mapping[str, Callable[..., Any]]):
    """The functions behind op IDs, imported when first asked for.

    A host needs the function to run a step again or to read its output's role,
    but only for the steps it has recorded. Importing every op in the
    collection when a session starts would be paying for all of them to use a
    few; a spec whose module will not import is discovery's business and is
    simply absent here.
    """

    def __init__(self, specs: list[OpSpec] | None = None) -> None:
        self._given = specs
        self._specs: dict[str, OpSpec] | None = None
        self._found: dict[str, Callable[..., Any]] = {}

    def _by_name(self) -> dict[str, OpSpec]:
        if self._specs is None:
            specs = self._given if self._given is not None else discover()[0]
            self._specs = {spec.name: spec for spec in specs}
        return self._specs

    def __getitem__(self, name: str) -> Callable[..., Any]:
        if name not in self._found:
            spec = self._by_name()[name]
            try:
                self._found[name] = resolve(spec)
            except Exception as exc:
                _log.warning("Could not import %s", name)
                raise KeyError(name) from exc
        return self._found[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_name())

    def __len__(self) -> int:
        return len(self._by_name())


class ViewerDisplay:
    """A napari viewer, as a host's display.

    A step's layer is found by the link in its metadata, never by its name: a
    name is a label the user is free to change, and renaming a layer must
    change nothing about what it is.
    """

    def __init__(self, viewer: napari.viewer.Viewer) -> None:
        self._viewer = viewer

    def layers(self) -> list[Any]:
        return list(self._viewer.layers)

    def show(
        self,
        name: str,
        data: Any,
        *,
        step: str,
        role: Role | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Put a step's output where the viewer can see it.

        The layer already drawing this step is given the new data, so running
        a step again does not pile up layers beside the first. A step with no
        layer yet gets one, of the type its role calls for -- a labels output
        lands in a labels layer, not an image one.

        A scalar output has no layer to be. The recorder keeps it; the viewer
        simply does not draw it.
        """
        if not hasattr(data, "shape"):
            _log.info("%s produced %r, which is not an array to draw", step, data)
            return None
        output = ((metadata or {}).get(SKOP) or {}).get("output")
        existing = self.layer_for(step, output)
        if existing is not None:
            existing.data = data
            existing.metadata.update(metadata or {})
            return existing
        args: dict[str, Any] = {"name": name, "metadata": dict(metadata or {})}
        return self._viewer.add_layer(
            Layer.create(data, args, layer_type_for_role(role) or "image")
        )

    def layer_for(self, step: str, output: str | None = None) -> Any:
        """The layer drawing a step, or one of its outputs, if there is one."""
        for layer in self._viewer.layers:
            link = layer.metadata.get(SKOP)
            if link is None or link.get("step") != step:
                continue
            # A link made by ``show`` names no output, and matches any.
            if output is None or link.get("output", output) == output:
                return layer
        return None


class Session:
    """What a viewer has recorded, and the runner everything is run with.

    Parameters
    ----------
    viewer
        Whose layers are the session's inputs, and where results are drawn.
    runner
        The scikit-ops runner; one is made when none is given.
    specs, registry
        The ops the session can name. Discovery supplies them unless a test
        says otherwise.
    root
        Where the store keeps its arrays and log. A temporary directory when
        none is given, which is the only place that is safe to write without
        being told, and so the log is gone with the machine's next clean-up.
    """

    #: Emitted, on the GUI thread, whenever what the session knows changed: a
    #: run was recorded or failed, a layer arrived or went, a layer was edited.
    #: A view of the session redraws on it instead of asking when to.
    changed = Signal()

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        *,
        runner: Runner | None = None,
        specs: list[OpSpec] | None = None,
        registry: Mapping[str, Callable[..., Any]] | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.viewer = viewer
        # napari constructs the panels, so there is no call site to pass
        # debug=True at; SKOP_NAPARI_DEBUG is the way in. It echoes the
        # worker's own stdout and stderr, which is the only view of what the
        # op is doing inside its own process.
        self.runner = (
            runner
            if runner is not None
            else Runner(debug=bool(os.environ.get("SKOP_NAPARI_DEBUG")))
        )
        self.store = SimpleStore(root or tempfile.mkdtemp(prefix="skop-napari-"))
        # The recorder's inner runner is the real one, so an op that names an
        # environment runs in it. Nothing here calls an op directly.
        self.recorder = Recorder(self.store, self.runner)
        self.listener = RecordingRunner(self.recorder, step_for=step_name)
        self.display = ViewerDisplay(viewer)
        self.host = Host(
            self.recorder,
            self.display,
            registry if registry is not None else OpRegistry(specs),
        )
        # So that an op ID a recipe holds can be turned back into its op.
        self.recorder.registry = self.host.registry
        self._recording = False
        #: Root layers painted on, waiting for the strokes to stop. A stroke
        #: reports on every mouse move, and a root is stored whole.
        self._painted_roots: dict[int, Any] = {}
        self._settle = QTimer()
        self._settle.setSingleShot(True)
        self._settle.setInterval(400)
        self._settle.timeout.connect(self._redeclare_painted)

        viewer.layers.events.inserted.connect(self._inserted)
        viewer.layers.events.removed.connect(self._removed)
        for layer in viewer.layers:
            self._watch(layer)

    # -- recording -------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._recording

    @recording.setter
    def recording(self, on: bool) -> None:
        if on == self._recording:
            return
        self._recording = on
        if on:
            # Whatever was opened while nothing was listening is an input too.
            self.declare_roots()
        self.changed.emit()

    @property
    def executor(self) -> Any:
        """What to run an op with: the runner, or the runner being listened to.

        Both are called the same way, so a caller does not need to know.
        """
        return self.listener if self._recording else self.runner

    def declare_roots(self) -> None:
        """Take every layer the session has not seen as a root.

        Also notices an input that moved while nothing was recording, so the
        steps that read it say so.
        """
        declared = self.host.sync()
        if declared:
            _log.info("Declared roots: %s", ", ".join(declared))
        self.changed.emit()

    def bind_root(self, name: str, layer: Any) -> None:
        """Make a layer the source of a root: choosing another input.

        The steps that read the root are stale from here on, which is the
        point -- they were made from something else.
        """
        if self.host.layer_of(name) is layer:
            return
        self.host.bind_root(name, layer)
        self.changed.emit()

    # -- drawing a recorded run ------------------------------------------

    def draw(
        self,
        step: str,
        output: str,
        data: Any,
        layer_type: str,
        args: dict[str, Any],
    ) -> Any:
        """Draw one recorded output as the layer that stands for its step.

        The layer already drawing the step is given the new data, when it is
        the same kind of layer: running a step again is a new run of the same
        step, and its picture should move with it rather than pile up beside
        the old one. A different kind of layer -- the user changed how masks
        are shown -- replaces it.

        The link goes in with the layer rather than being added after, so
        that no one sees a layer arrive that is not yet known for what it is.
        """
        existing = self.display.layer_for(step, output)
        link = self.host.link(step, output, data)
        if link is None:
            raise KeyError(f"{step!r} has no recorded output {output!r}")

        with self.host.writing():
            if existing is not None and (type(existing).__name__.lower() == layer_type):
                existing.data = data
                existing.metadata[SKOP] = link
                layer = existing
            else:
                if existing is not None:
                    self.viewer.layers.remove(existing)
                metadata = {**args.get("metadata", {}), SKOP: link}
                layer = self.viewer.add_layer(
                    Layer.create(data, {**args, "metadata": metadata}, layer_type)
                )
        self.changed.emit()
        return layer

    # -- bringing steps up to date ---------------------------------------

    def produce(self, targets: list[str]) -> list[str]:
        """Bring the targets up to date through the runner, and say what ran.

        Blocks for as long as the ops take, so it belongs on a worker thread.
        The listener is made ambient so that a workflow that is run again
        records what it calls, as it did the first time.
        """
        from skop import ambient

        with ambient(self.listener):
            return self.host.produce(targets)

    # -- keeping up with the viewer --------------------------------------

    def _inserted(self, event: Any) -> None:
        """A layer appeared: watch it, and take it as a root.

        Every layer the viewer holds that the host has not seen is an input,
        which is what makes "open an image, then run an op" work with no
        declaration step in between. A layer drawn by a run already carries
        its link and is left alone.
        """
        self._watch(event.value)
        if self._recording:
            self.declare_roots()

    def _removed(self, event: Any) -> None:
        if self._recording:
            gone = self.host.layer_removed(event.value)
            if gone:
                _log.info("Input %r has no layer any more", gone)
        self.changed.emit()

    def _watch(self, layer: Any) -> None:
        """Listen for a layer's data changing by hand, or being painted on."""
        layer.events.data.connect(self._data_changed)
        # Only a Labels layer paints, and napari does not report a stroke as a
        # change of data: the array is changed in place.
        paint = getattr(layer.events, "paint", None)
        if paint is not None:
            paint.connect(self._painted)

    def _painted(self, event: Any) -> None:
        if not self._recording:
            return
        layer = event.source
        try:
            mattered = self.host.painted(layer)
        except Exception:  # noqa: BLE001 -- an event handler must not raise.
            _log.exception("Could not read the paint on %r", layer)
            return
        if not mattered:
            return
        if self.host.is_root(layer):
            self._painted_roots[id(layer)] = layer
            self._settle.start()
        else:
            self.changed.emit()

    def _redeclare_painted(self) -> None:
        """The strokes stopped: store each painted input as a new version."""
        painted, self._painted_roots = self._painted_roots, {}
        for layer in painted.values():
            try:
                self.host.declare_root(layer)
            except Exception:  # noqa: BLE001 -- see _painted.
                _log.exception("Could not store the paint on %r", layer)
        self.changed.emit()

    def _data_changed(self, event: Any) -> None:
        """Note a hand edit, or a root moving."""
        if not self._recording:
            return
        try:
            mattered = self.host.data_changed(event.source)
        except Exception:  # noqa: BLE001 -- an event handler must not raise.
            _log.exception("Could not read the change to %r", event.source)
            return
        if mattered:
            self.changed.emit()


#: Weakly held: a session lives as long as a panel is using it, and no longer.
#: Held strongly it would keep its viewer alive for the life of the process.
_sessions: weakref.WeakValueDictionary[int, Session] = weakref.WeakValueDictionary()


def session_for(viewer: napari.viewer.Viewer, **options: Any) -> Session:
    """The viewer's session, made the first time anything asks for it.

    The panels are built by napari, one at a time and in any order, and they
    have to meet in the same session: that is what lets the Record panel show
    what the Ops panel ran. ``options`` only matter to whoever asks first.
    """
    found = _sessions.get(id(viewer))
    if found is None:
        found = _sessions[id(viewer)] = Session(viewer, **options)
    return found
