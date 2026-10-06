"""Running an op without freezing napari.

``Runner.run`` blocks until the op finishes, and the op is doing real work in
another process -- seconds for a threshold, minutes for a 3D model. So the
call goes onto a worker thread, and everything coming back from it has to be
marshalled onto the GUI thread before it touches a widget.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from napari.qt.threading import FunctionWorker, create_worker
from superqt.utils import ensure_main_thread

from skop import OpSpec, Runner


def resolve(spec: OpSpec) -> Callable:
    """Get the op function a spec describes."""
    return getattr(importlib.import_module(spec.module), spec.function)


class OpRun:
    """One in-flight op call.

    Holds the Appose task so that ``cancel`` has something to act on. The
    task arrives from the worker thread partway through, hence the guard: a
    Cancel pressed in the first instants has nothing to cancel yet.
    """

    def __init__(
        self,
        runner: Runner,
        spec: OpSpec,
        kwargs: dict[str, Any],
        *,
        plans: dict[str, Any] | None = None,
        on_progress: Callable[[str | None, int | None, int | None], None],
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        on_finish: Callable[[], None],
    ) -> None:
        self.spec = spec
        self._task: Any = None
        self._canceled = False
        self._live = True

        fn = resolve(spec)

        # Progress arrives on an Appose listener thread; ensure_main_thread
        # bounces it back to Qt, which means delivery happens later than the
        # event. By then the run may be over -- so check on arrival, not on
        # dispatch, and drop anything that outlived its run.
        def relay(event: Any) -> None:
            if self._live:
                on_progress(event.message, event.current, event.maximum)

        report = ensure_main_thread(relay)

        def finished() -> None:
            self._live = False
            on_finish()

        def work() -> Any:
            return runner.run(
                fn,
                kwargs,
                plans=plans or None,
                on_progress=report,
                on_start=self._started,
            )

        # NB: the callbacks go through _connect rather than being connected
        # afterwards, for two reasons. create_worker starts the thread
        # immediately unless told not to, so connecting after the call races
        # a fast op; and superqt re-raises errors into the event loop unless
        # an 'errored' handler was supplied this way, which would turn every
        # failed op into a napari traceback popup on top of our own report.
        self.worker: FunctionWorker = create_worker(
            work,
            _start_thread=False,
            _connect={
                "returned": on_done,
                "errored": on_error,
                "finished": finished,
            },
        )

    def detach(self) -> None:
        """Stop relaying progress, e.g. because the panel is going away."""
        self._live = False

    def _started(self, task: Any) -> None:
        self._task = task
        # A cancel that landed before the task existed still has to take.
        if self._canceled:
            task.cancel()

    def start(self) -> None:
        self.worker.start()

    def cancel(self) -> None:
        """Ask the op to stop.

        Cooperative: the op stops at its next ``skop.cancel_requested()``
        check, and returns whatever it had computed by then. An op that never
        checks runs to completion.
        """
        self._canceled = True
        if self._task is not None:
            self._task.cancel()


def outputs_of(spec: OpSpec, result: Any) -> dict[str, Any]:
    """Label an op's return value with its output names.

    ``Runner.run`` hands back exactly what the op returned -- a bare value
    when the op declared a plain return type, a NamedTuple when it declared
    one -- which is pleasant to call and useless to iterate generically. The
    spec says what the pieces are called.

    The NamedTuple case is tested first, and on its fields rather than on how
    many there are. An op may declare a NamedTuple with a single field, to
    give its one output a name better than "result"; counting outputs cannot
    tell that from a bare return, and would label the tuple itself as the
    output -- handing napari a tuple where an array belongs.
    """
    names = tuple(o.name for o in spec.outputs)
    if not names:
        return {}
    if getattr(result, "_fields", None) == tuple(names):
        return dict(zip(names, result, strict=True))
    if len(names) == 1:
        return {names[0]: result}
    return dict(zip(names, result, strict=False))
