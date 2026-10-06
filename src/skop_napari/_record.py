"""The Record panel: what has been run, and what to do about it.

It runs nothing. The Ops and Workflows panels run ops, as they always did, and
while this panel is recording they run them through the session's listening
runner (``_session``) -- so a call made anywhere, in any environment, is written
down on its way past, and this panel is where the writing shows. It lists the
steps, says which have gone stale or been edited by hand, and brings the ones
you tick up to date.

What a host is, and what it promises, is in ``_host``: that module imports no
napari and is tested without a viewer. What a session is -- the one runner and
store a viewer's panels share -- is in ``_session``. This module is the two
things neither can supply for itself, something to draw in and somewhere to
press buttons.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import napari.viewer
from magicgui.widgets import CheckBox, Container, Label, PushButton
from napari.qt.threading import create_worker
from napari.utils.notifications import show_error

from skop import OpSpec
from skop_visual import EDITED, SIGNALS

from ._host import SKOP
from ._session import Session, session_for

_log = logging.getLogger("skop_napari")

#: The widths the table pads its columns to, and the header that names them.
#: Fixed because a row is one string: magicgui has no table, and the point of
#: the wording is that a column of statuses can be read down.
_STEP_WIDTH = 22
_OP_WIDTH = 34
_HEADER = f"{'step':<{_STEP_WIDTH}}{'op':<{_OP_WIDTH}}state"


def _short_op(op: str) -> str:
    """An op ID, shortened to what tells two of them apart.

    An ID is ``module:function`` and the module is long. What is worth showing
    is the namespace the op lives in -- 'mask: detect' next to 'image: smooth'
    -- so the module's own name is dropped when it repeats the function.
    """
    module, _, function = op.rpartition(":")
    tail = module.rsplit(".", 1)[-1]
    return function if tail == function else f"{tail}:{function}"


def _row_text(node: Any) -> str:
    """One row of the table, as text.

    Text rather than a painted cell: a background colour is a code the reader
    has to learn, and the words are already written. The state is the same
    signal the graph diagram uses, so the two read alike.
    """
    return (
        f"{node.step:<{_STEP_WIDTH}}{_short_op(node.op):<{_OP_WIDTH}}"
        f"{SIGNALS[node.badge]}"
    )


def _fixed_font() -> Any:
    """A monospace font, so the table's columns line up.

    Every failure here is cosmetic and every one of them is better than a
    panel that will not open, so this never raises. The enum is scoped in Qt6
    and unscoped in Qt5, hence the pair of getattr calls rather than one.
    """
    try:
        from qtpy.QtGui import QFontDatabase

        kind = getattr(QFontDatabase, "SystemFont", QFontDatabase)
        return QFontDatabase.systemFont(kind.FixedFont)
    except Exception:  # noqa: BLE001 -- a ragged table beats no panel.
        return None


class RecordPanel(Container):
    """Shows what the session recorded, and brings steps up to date."""

    def __init__(
        self,
        napari_viewer: napari.viewer.Viewer | None = None,
        specs: list[OpSpec] | None = None,
        registry: dict[str, Any] | None = None,
        root: str | Path | None = None,
    ) -> None:
        # NB: named napari_viewer for the reason the Ops panel gives -- it is
        # the only check napari makes that survives a postponed annotation.
        if napari_viewer is None:
            raise ValueError(
                "The Record panel shows what a viewer's session recorded, and "
                "needs the viewer. napari supplies one to a parameter named "
                "'napari_viewer'."
            )
        self._viewer = napari_viewer
        self._session: Session = session_for(
            napari_viewer, specs=specs, registry=registry, root=root
        )
        self._host = self._session.host
        self._recorder = self._session.recorder
        #: Which steps are ticked as targets. Kept here rather than read back
        #: out of the widgets, so rebuilding the table does not lose it.
        self._targets: set[str] = set()
        self._font = _fixed_font()
        #: The worker bringing steps up to date, while there is one.
        self._producing: Any = None

        self._recording = CheckBox(
            value=True,
            text="Record what is run",
            tooltip=(
                "While this is on, ops run from the Ops and Workflows panels "
                "are written down. They run exactly as they would otherwise."
            ),
        )
        self._roots_heading = Label(value="Roots")
        self._roots = Label(value="")
        self._declare = PushButton(text="Declare the viewer's layers")

        self._runs_heading = Label(
            value="Recorded",
            tooltip=(
                f"The log of what was run is written to {self._session.store.log_path}."
            ),
        )
        self._header = Label(value=_HEADER)
        self._table = Container(labels=False)
        self._produce = PushButton(text="Produce")

        self._status = Label(value="")

        super().__init__(
            widgets=[
                self._recording,
                self._roots_heading,
                self._roots,
                self._declare,
                self._runs_heading,
                self._header,
                self._table,
                self._produce,
                self._status,
            ],
            labels=False,
        )

        self._recording.changed.connect(self._record_toggled)
        self._declare.changed.connect(self._declare_roots)
        self._produce.changed.connect(self._produce_)
        # Whatever ran an op -- the Ops panel, a script -- is told about here
        # only by the session saying its picture changed.
        self._session.changed.connect(self._refresh)

        self._session.recording = self._recording.value
        self._refresh()

    # -- recording -------------------------------------------------------

    def _record_toggled(self, on: bool) -> None:
        self._session.recording = on
        self._status.value = (
            "Recording." if on else "Not recording: ops run, and are forgotten."
        )

    # -- roots -----------------------------------------------------------

    def _declare_roots(self) -> None:
        """Take every layer the host has not seen as a root.

        Idempotent, and safe to press twice: a layer already carrying its link
        is left alone, which is also what makes a duplicate of a layer a view
        of the same step rather than a new input.
        """
        self._session.declare_roots()

    def _roots_text(self) -> str:
        names = [
            layer.name
            for layer in self._host.display.layers()
            if (layer.metadata.get(SKOP) or {}).get("step", "missing") is None
        ]
        if not names:
            return "Nothing yet. An op needs a layer to read."
        return ", ".join(names)

    # -- what was recorded -----------------------------------------------

    def _refresh(self) -> None:
        """Rebuild the table from the graph, which is the only account of it.

        Read back from the log rather than remembered: a panel that kept its
        own idea of what had been run is a second source of truth, and the one
        that goes wrong.
        """
        try:
            self._rebuild()
        except RuntimeError:
            # The panel was closed but the session outlived it, so this is
            # still arriving at widgets whose Qt objects are gone.
            self._session.changed.disconnect(self._refresh)

    def _rebuild(self) -> None:
        view = self._host.view()
        steps = {node.step for node in view.nodes}
        self._targets &= steps

        self._header.native.setFont(self._font) if self._font else None
        self._table.clear()
        for node in view.nodes:
            row = Container(
                widgets=[
                    Label(value=_row_text(node)),
                    CheckBox(value=node.step in self._targets, tooltip="Produce this"),
                ],
                labels=False,
            )
            if self._font:
                for widget in row:
                    widget.native.setFont(self._font)
            row[1].changed.connect(partial(self._tickle, node.step))
            if EDITED in node.flags:
                # Only an edited step needs these, so they are only built for
                # one: a button that does nothing is worse than no button.
                row.extend(self._edit_buttons(node.step))
            self._table.append(row)

        self._roots.value = self._roots_text()
        self._produce.enabled = bool(self._targets) and self._producing is None

    def _edit_buttons(self, step: str) -> list[Any]:
        revert = PushButton(text="Revert")
        revert.changed.connect(partial(self._revert, step))
        keep = PushButton(text="Keep the edit")
        keep.changed.connect(partial(self._keep, step))
        return [revert, keep]

    def _tickle(self, step: str, ticked: bool) -> None:
        if ticked:
            self._targets.add(step)
        else:
            self._targets.discard(step)
        self._produce.enabled = bool(self._targets) and self._producing is None
        self._status.value = self._targets_line()

    def _targets_line(self) -> str:
        if not self._targets:
            return "Tick the steps to bring up to date, then Produce."
        return f"Targets: {', '.join(sorted(self._targets))}"

    def _produce_(self) -> None:
        """Bring the ticked steps up to date, and draw what moved.

        Produce runs the backward closure of the targets and nothing else
        (decision 4), so a step whose inputs have not moved is left alone and
        costs nothing. It runs the ops through the session's runner, which may
        mean building an environment, so it happens on a worker: the viewer
        stays usable, and the layers are drawn when it is done.
        """
        if not self._targets or self._producing is not None:
            self._status.value = self._targets_line()
            return
        targets = sorted(self._targets)
        self._produce.enabled = False
        self._status.value = f"Producing {', '.join(targets)}..."
        # NB: connected here rather than afterwards, for the reason the Ops
        # panel gives: the thread starts at once, and superqt re-raises into
        # the event loop unless an 'errored' handler was supplied this way.
        self._producing = create_worker(
            partial(self._session.produce, targets),
            _start_thread=False,
            _connect={
                "returned": self._produced,
                "errored": partial(self._produce_failed, targets),
                "finished": self._produce_finished,
            },
        )
        self._producing.start()

    def _produced(self, ran: list[str]) -> None:
        for step in ran:
            try:
                self._host.show(step)
            except Exception:  # noqa: BLE001 -- a scalar output has no layer.
                _log.info("Nothing to draw for %r", step)
        self._status.value = f"Ran {', '.join(ran)}." if ran else "Already up to date."

    def _produce_failed(self, targets: list[str], exc: Exception) -> None:
        _log.error("Could not produce %s", targets, exc_info=exc)
        self._status.value = f"{type(exc).__name__}: {exc}"
        show_error(str(exc))

    def _produce_finished(self) -> None:
        self._producing = None
        self._refresh()

    # -- the two ways an edit ends ---------------------------------------

    def _revert(self, step: str) -> None:
        """Put a step's recorded output back over its layer."""
        try:
            self._host.revert(step)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"{type(exc).__name__}: {exc}"
            return
        self._status.value = f"Put {step!r} back to what its run produced."
        self._refresh()

    def _keep(self, step: str) -> None:
        """Keep the hand-made pixels, and stop the step claiming to reproduce.

        Accepting adds a record like everything else: the run stays, and the
        mark after it says the pixels are the user's now.
        """
        try:
            self._recorder.accept_edit(step)
        except Exception as exc:  # noqa: BLE001
            self._status.value = f"{type(exc).__name__}: {exc}"
            return
        self._status.value = f"Kept the edit to {step!r}."
        self._refresh()
