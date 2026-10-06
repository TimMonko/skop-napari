"""The recording host: layers in, runs out, layers back.

A host owns a recorder and a *display* -- whatever is showing layers. The
recorder speaks only in ops and handles; the display knows only about names,
data and metadata. This module is what turns one into the other, and it imports
no napari, so everything it decides is exercised by tests that need no viewer.

A layer is a *view* of one step's output, or of a root. The link lives in
``layer.metadata[SKOP]``: a live dict, copied when a layer is created and
carried in ``as_layer_data_tuple``, so it survives a layer copy, and assigning
into it emits nothing, so our bookkeeping never looks like a user edit.

The link carries an id the host issues, and that id -- not the layer's name --
is what the host knows a layer by. A name is only a label, and renaming a layer
must change nothing. napari's ``layer.source`` cannot serve instead: ``Source``
has no identity field and is frozen.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from skop import OpSpec, Role
from skop_core import Handle, Recorder
from skop_visual import GraphView
from skop_visual import view as build_view

#: The key a layer's bookkeeping lives under, in ``layer.metadata``.
SKOP = "skop"

_log = logging.getLogger("skop_napari")


def _new_id() -> str:
    """An id for a layer, issued once and kept in its metadata."""
    return uuid.uuid4().hex[:12]


def _is_layer(value: Any) -> bool:
    """Whether a value is a layer rather than the array itself.

    A panel hands over the layer a user picked; an update hands over the array
    a step already read. Only a layer has metadata, so that is the tell.
    """
    return hasattr(value, "data") and hasattr(value, "metadata")


class UnknownLayer(ValueError):
    """A layer reached an op without this host having declared it."""


class UnknownOp(KeyError):
    """A step named an op this host has no entry for."""


class Layer(Protocol):
    """One layer, as much of it as a host needs to see."""

    name: str
    data: Any
    metadata: dict[str, Any]


class Display(Protocol):
    """Where layers are shown: everything the host asks of a viewer.

    Two methods, because that is the whole of what a host needs. A viewer that
    can list its layers and put one there is a host's display.

    ``show`` finds the layer that already draws ``step`` -- by the link in its
    metadata, never by its name -- and replaces the data, and the layer type if
    the role wants another one. It creates a layer named ``name`` when the step
    has none. It is the display that knows what a role means to a viewer: an op
    whose output says ``Role.labels`` gets a labels layer, not an image one.
    """

    def layers(self) -> Sequence[Layer]: ...

    def show(
        self,
        name: str,
        data: np.ndarray,
        *,
        step: str,
        role: Role | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Layer: ...


@dataclass(frozen=True)
class _View:
    """What the host remembers about one layer it linked."""

    #: The id the host issued the layer, and wrote into its metadata.
    id: str
    #: The step this layer draws, or None when the layer is a root.
    step: str | None
    handle: Handle
    #: The exact object the store was given or handed back, for identity.
    array: np.ndarray
    #: Which of the step's outputs this draws, when the step has several.
    output: str | None = None


class Host:
    """A recorder and a display, kept in step.

    Parameters
    ----------
    recorder
        Where runs are recorded and arrays are stored.
    display
        Where layers are shown.
    registry
        The ops this host can name, by ID -- ``OpSpec.name``, as ``discover()``
        reports it. Needed to resolve a step's op, and to read the role its
        output is drawn with. Kept as given rather than copied, so a registry
        that imports an op only when asked for it stays lazy.
    """

    def __init__(
        self,
        recorder: Recorder,
        display: Display,
        registry: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self.recorder = recorder
        self.display = display
        self.registry: Mapping[str, Callable[..., Any]] = (
            {} if registry is None else registry
        )
        #: Every layer the host linked, by the id it issued that layer.
        self._views: dict[str, _View] = {}
        #: The view that stands behind each root name right now. A name is a
        #: slot, not a layer: the layer in it can change (see ``bind_root``),
        #: and only the one that holds it is the root's source.
        self._owner: dict[str, str] = {}
        self._writing = 0

    # -- roots -----------------------------------------------------------

    def sync(self) -> list[str]:
        """Declare what is new, and notice roots that moved unobserved.

        Declaring is for layers nobody has seen. A layer that was seen and then
        changed while no one was listening -- recording was off, say -- is
        found here by the identity of its data, as an event would have found
        it. Only a layer holding a plain array is checked: a lazy one hands
        out a new object on every read, and would look moved every time.
        """
        declared = self.declare_roots()
        for layer in self.display.layers():
            if (
                self.linked(layer)
                and isinstance(layer.data, np.ndarray)
                and (found := self._view_of(layer)) is not None
                and found.step is None
            ):
                self.data_changed(layer)
        return declared

    def declare_roots(self) -> list[str]:
        """Declare every layer this session has not seen as a root.

        Call it when layers appear. Idempotent: a layer already carrying the
        link is left alone, which is also what makes a copy of a layer
        harmless.

        A layer whose data is not an array -- a shapes layer holds a list of
        vertex arrays -- cannot be stored, and is skipped rather than taking
        the layers after it down too. It stays undeclared, so an op handed it
        is refused in words by the recorder rather than recorded wrongly.
        """
        declared = []
        for layer in self.display.layers():
            if self.linked(layer):
                continue
            try:
                self.declare_root(layer)
            except Exception:  # noqa: BLE001 -- see above.
                _log.warning("Could not declare %r as a root", layer.name)
                continue
            declared.append(layer.name)
        return declared

    def linked(self, layer: Layer) -> bool:
        """Whether this host already knows what a layer is."""
        return SKOP in layer.metadata

    def declare_root(self, layer: Layer) -> Handle:
        """Store a layer's data as a root, under the name the layer holds.

        A layer is named by its own name the first time, and by that name from
        then on even if the layer is renamed: a root's name is what its readers
        know it by, and renaming a layer must not cut them off from it.

        The array handed on is the one remembered, never a fresh read of the
        layer: the store keys identity off the exact object, so a second
        materialisation would look like a root nobody declared.

        Declaring a layer that is already a root stores a new version under the
        same name and keeps the id. That is the root moving, and whoever read
        the old version is stale from then on.
        """
        view_id = self._id_of(layer) or _new_id()
        return self._store_root(layer, view_id, self._root_name(layer, view_id))

    def bind_root(self, name: str, layer: Layer) -> Handle:
        """Make a layer the source of a root, in place of whichever was.

        This is what choosing another layer for an input means. The root keeps
        its name, so everything that read it is unchanged -- and stale, because
        what stands behind the name is new. The layer that held the name before
        keeps its link but not the name, and takes a fresh one of its own the
        next time it changes.
        """
        view_id = self._id_of(layer) or _new_id()
        previous = (layer.metadata.get(SKOP) or {}).get("root")
        handle = self._store_root(layer, view_id, name)
        if previous and previous != name and self._owner.get(previous) == view_id:
            # The layer used to be a root of its own, and is not any more.
            del self._owner[previous]
            self.recorder.root_gone(previous)
        return handle

    def _store_root(self, layer: Layer, view_id: str, name: str) -> Handle:
        array = np.asarray(layer.data)
        handle = self.recorder.root(name, array)
        self._owner[name] = view_id
        layer.metadata[SKOP] = {
            "id": view_id,
            "step": None,
            "handle": handle,
            "run": None,
            "root": name,
        }
        self._views[view_id] = _View(view_id, None, handle, array)
        return handle

    def _root_name(self, layer: Layer, view_id: str) -> str:
        """The name a layer's root goes by: the one it holds, or a new one."""
        held = (layer.metadata.get(SKOP) or {}).get("root")
        if held is not None and self._owner.get(held, view_id) == view_id:
            return held
        name, count = layer.name, 1
        while self._owner.get(name, view_id) != view_id:
            count += 1
            name = f"{layer.name} ({count})"
        return name

    def is_root(self, layer: Layer) -> bool:
        """Whether a layer is an input, rather than a view of a step."""
        found = self._view_of(layer)
        return found is not None and found.step is None

    def root_role(self, root: str) -> Role | None:
        """What the ops that read a root take it to be, when they agree.

        Read off the parameters that consume it, so a chooser can offer only
        the layers that fit: an input a labels parameter reads is not offered
        an image. None when nothing has read it, or its readers disagree.
        """
        roles: set[Role | None] = set()
        graph = self.recorder.graph()
        for run in graph.runs:
            fn = self.registry.get(run.op)
            if fn is None:
                continue
            by_name = {p.name: p for p in OpSpec.from_op(fn).params}
            for param, handle in run.inputs.items():
                found = graph.root_of(handle)
                if found is not None and found.name == root and param in by_name:
                    roles.add(by_name[param].role)
        return roles.pop() if len(roles) == 1 else None

    def layer_of(self, root: str) -> Layer | None:
        """The layer standing behind a root, or None when there is none."""
        view_id = self._owner.get(root)
        if view_id is None:
            return None
        for layer in self.display.layers():
            if self._id_of(layer) == view_id:
                return layer
        return None

    def layer_removed(self, layer: Layer) -> str | None:
        """Note that a layer left the viewer, and say which root it took.

        The root is recorded as gone, so what read it can say so. The data is
        kept: nothing about the past is lost, only where it came from. A layer
        that was only a *view* of a step -- a result, rather than an input --
        leaves the log alone, since the step's output is still in the store.
        """
        link = layer.metadata.get(SKOP)
        if not link:
            return None
        view_id = link.get("id")
        if any(self._id_of(other) == view_id for other in self.display.layers()):
            return None  # a copy of it is still there
        self._views.pop(view_id, None)
        name = link.get("root")
        if link.get("step") is None and name and self._owner.get(name) == view_id:
            del self._owner[name]
            self.recorder.root_gone(name)
            return name
        return None

    def painted(self, layer: Layer) -> bool:
        """Note a paint stroke, and say whether it needs acting on.

        Painting changes an array in place and emits ``paint`` rather than
        ``data``, so identity says nothing about it. A step's layer is marked
        edited once, however many strokes follow. A root's layer is reported
        for the caller to declare again when the strokes are done: storing a
        whole array per mouse move is not something to do here.
        """
        if self._writing:
            return False
        found = self._view_of(layer)
        if found is None:
            return False
        if found.step is None:
            return True
        if self.recorder.graph().is_edited(found.step):
            return False
        self.recorder.mark_edited(found.step)
        return True

    def link(self, step: str, output: str, shown: np.ndarray) -> dict[str, Any] | None:
        """The metadata that makes a layer a view of a step's current output.

        For a caller that draws the layer itself -- a panel that has already
        run an op, and knows how it wants the result shown -- where :meth:`show`
        is for a host that draws it. Put the dict in the layer's metadata under
        :data:`SKOP` when the layer is created, so the link is there before
        anyone is told the layer exists.

        ``shown`` is the object the layer will hold, and what an edit is
        measured against: a layer whose data is still that object has not been
        touched. None when the step has no such output.
        """
        run = self.recorder.graph().current(step)
        handle = None if run is None else run.outputs.get(output)
        if run is None or handle is None:
            return None
        existing = self._view_of_step(step, output)
        view_id = existing.id if existing is not None else _new_id()
        self._views[view_id] = _View(view_id, step, handle, shown, output)
        return {
            "id": view_id,
            "step": step,
            "handle": handle,
            "run": run.id,
            "output": output,
        }

    def array_for(self, layer: Layer) -> np.ndarray:
        """The stored array a layer stands for, for handing to an op.

        A layer this host linked gives back the array the store was handed, by
        identity. When the layer's data is no longer that exact object -- a lazy
        layer materialised a second time, or a layer type that hands out a
        fresh array -- the handle in the layer's own link is used instead, since
        nothing says the data changed.

        An *edited* layer is the exception, and the point of the rule: handing
        back the recorded array would quietly discard the edit, so this refuses
        and the user accepts or reverts first.
        """
        found = self._view_of(layer)
        if found is None:
            raise UnknownLayer(
                f"layer {layer.name!r} is not one this host declared. Declare "
                f"the layers the viewer holds, with host.declare_roots()."
            )
        if found.array is layer.data:
            return found.array
        if found.step is None or self.recorder.graph().is_edited(found.step):
            raise UnknownLayer(
                f"the data in {layer.name!r} is not what was recorded for "
                f"{found.step!r}. Accept the edit, or revert to what the run "
                f"produced, before running anything on it."
            )
        return self.recorder.store.get(found.handle)

    # -- running ---------------------------------------------------------

    def run(self, step: str, op_id: str, args: dict[str, Any]) -> Any:
        """Record one op call, then draw what it made.

        ``args`` maps parameter names to values. An array parameter takes the
        layer the user picked, which the host resolves to data the store
        already holds; everything else is a parameter, and must be JSON for the
        recorder to accept it.
        """
        resolved = {
            name: self.array_for(value) if _is_layer(value) else value
            for name, value in args.items()
        }
        result = self.recorder.run(self._fn(op_id), step, resolved)
        self.show(step)
        return result

    def update(self, step: str) -> Any:
        """Run a stale step again, taking each input from where it now is."""
        return self.recorder.update(self._fn(self._current(step).op), step)

    def rerun(self, step: str) -> Any:
        """Run a step again from the run it is on: its inputs, its params."""
        return self.recorder.rerun(self._fn(self._current(step).op), step)

    def produce(self, targets: list[str]) -> list[str]:
        """Bring the targets' closure up to date, and say what that ran."""
        return self.recorder.produce(list(targets), self.registry)

    def show(self, step: str) -> list[Layer]:
        """Draw a step's current run as its layer, linked in the metadata.

        The role comes from the op, so a labels output lands in a labels layer
        rather than an image one. An op with several outputs wants one layer
        each, which is not built yet.
        """
        run = self.recorder.graph().current(step)
        if run is None:
            raise KeyError(f"no successful run of {step!r} to draw")
        if len(run.outputs) > 1:
            raise NotImplementedError(
                f"{step!r} has {len(run.outputs)} outputs; drawing more than "
                f"one output of a step is not built yet"
            )
        existing = self._view_of_step(step)
        view_id = existing.id if existing is not None else _new_id()
        drawn = []
        with self.writing():
            for name, handle in run.outputs.items():
                array = self.recorder.store.get(handle)
                layer = self.display.show(
                    step,
                    array,
                    step=step,
                    role=self.role_of(run.op, name),
                    metadata={
                        SKOP: {
                            "id": view_id,
                            "step": step,
                            "handle": handle,
                            "run": run.id,
                        }
                    },
                )
                self._views[view_id] = _View(view_id, step, handle, array)
                drawn.append(layer)
        return drawn

    def revert(self, step: str) -> list[Layer]:
        """Put a step's recorded output back over its layer, and unmark it.

        The edit is not deleted -- the log is append-only -- but a revert mark
        supersedes it, so the step stops being edited and nothing downstream
        stays affected.
        """
        edited = self.recorder.graph().is_edited(step)
        drawn = self.show(step)
        if edited:
            self.recorder.mark_reverted(step)
        return drawn

    def role_of(self, op_id: str, output: str) -> Role | None:
        """The role an op gives one of its outputs, when the op is known."""
        fn = self.registry.get(op_id)
        if fn is None:
            return None
        for found in OpSpec.from_op(fn).outputs:
            if found.name == output:
                return found.role
        return None

    # -- what is drawn ---------------------------------------------------

    def view(self, targets: list[str] | None = None) -> GraphView:
        """The graph, as a panel or a Mermaid renderer draws it."""
        return build_view(self.recorder.graph(), self.recorder.store, targets)

    # -- keeping in step with the viewer ---------------------------------

    @contextmanager
    def writing(self) -> Iterator[None]:
        """Bracket the host's own writes, so they are not read as edits."""
        self._writing += 1
        try:
            yield
        finally:
            self._writing -= 1

    def data_changed(self, layer: Layer) -> bool:
        """Note that a layer's data changed, and say whether it mattered.

        A change outside :meth:`writing` is a hand edit: the step is marked,
        and everything downstream is affected, derived on read rather than
        stored. A change inside the bracket is ours and means nothing. Identity
        is compared, never contents, so re-assigning the same array is not an
        edit and a paint that reports twice marks once.

        A root is not a step, so a hand edit to one is not a mark. It is the
        root moving: a new version goes into the store, and whoever read the
        old one is stale.
        """
        if self._writing:
            return False
        found = self._view_of(layer)
        if found is None or found.array is layer.data:
            return False
        if found.step is None:
            self.declare_root(layer)
            return True
        self.recorder.mark_edited(found.step)
        return True

    # -- what the host knows a layer by ----------------------------------

    def _view_of(self, layer: Layer) -> _View | None:
        view_id = self._id_of(layer)
        return None if view_id is None else self._views.get(view_id)

    def _view_of_step(self, step: str, output: str | None = None) -> _View | None:
        for view in self._views.values():
            if view.step == step and (output is None or view.output in (None, output)):
                return view
        return None

    @staticmethod
    def _id_of(layer: Layer) -> str | None:
        """The id the host issued a layer, read off the layer's own link."""
        link = layer.metadata.get(SKOP)
        return None if link is None else link.get("id")

    def _fn(self, op_id: str) -> Callable[..., Any]:
        """The function an op ID names, or a clear refusal.

        A recorder resolves no IDs -- a recipe does, through a registry -- so
        the host is what turns an op's name back into something callable.
        """
        fn = self.registry.get(op_id)
        if fn is None:
            raise UnknownOp(f"{op_id!r} is not an op this host can run")
        return fn

    def _current(self, step: str):
        run = self.recorder.graph().current(step)
        if run is None:
            raise KeyError(f"no successful run of {step!r}")
        return run
