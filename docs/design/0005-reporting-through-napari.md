# 0005 — Reporting: errors, build progress, and silence

Three ways a run can produce something other than a layer, and one bug that
showed why all three matter.

## Errors go to napari, not into the panel

The first implementation put a failure message in a `LineEdit` in the panel.
That is wrong in a specific way: an op fails **in another process, running
another interpreter**, and the interesting part of the failure is that
process's traceback. A one-line field cannot show it.

```python
def _on_error(self, exc: Exception) -> None:
    _log.error("Op %s failed", self.spec.name, exc_info=exc)
    notification_manager.receive_error(type(exc), exc, exc.__traceback__)
```

napari's notification carries the traceback, keeps it in the notification
history so a notification that has come and gone is still recoverable, and
looks like every other napari error. None of which the panel could do without
reimplementing all of it.

The message text is worth knowing for tests: it reads `Task failed:
Traceback...`. The exception *type name* is not in it, so asserting on
`"TaskException"` fails while asserting on `"Task failed"` and on frames from
the worker's own traceback (`toy.py`, `image * factor`) succeeds.

**Test isolation:** `notification_manager.records` accumulates across the whole
session, so a test asserting on its length must slice off what was already
there. `tests/test_panel.py` has an `errors_raised()` context manager for this;
without it, the second error-raising test sees two errors.

## Build progress drives the same bar

Environment building happens inside `run()` and is by far the slowest part of a
first run, so skop's three
[build subscriptions](https://github.com/apposed/scikit-ops/blob/main/docs/design/0004-build-feedback.md)
feed the panel's existing progress bar. A fresh build looks like:

```
Preparing environment: probe          (indeterminate)
Installing conda packages             0/30
Installing conda packages             30/30
Done                                  1/1
✔ The default environment has been installed.
```

The `Preparing environment: <id>` label is set by the panel itself before
anything is subscribed. There is no "build started" event, and on a cold cache
pixi can sit silent for a while; without that label the panel looks hung.

Two consequences of skop's design note, realized here:

- **`subscribe_error` is stderr, not failure.** It is wired to `_on_build_text`,
  the same handler as `subscribe_output` — *not* to the error path. Pixi writes
  its success message there. Routing it to `receive_error` turns every
  successful build into an error notification, which is precisely what the
  first version did.
- **Subscribing to progress turns on `-vv`.** So the output/error channels
  carry pixi's entire debug log. `_on_build_text` logs all of it to the
  `napari_skopkit` logger and shows only lines that do not match `_LOG_RECORD`
  (`^\s*(TRACE|DEBUG|INFO|WARN|WARNING|ERROR)\b`). Detail stays recoverable;
  the progress bar stays readable.

All three subscribers are wrapped in `ensure_main_thread`, since environments
are built on the worker thread.

## The bug: every output silently vanished

Symptom: run an op in napari, watch the progress bar reach 100%, and then
nothing. No layer, no error, no stdout.

Cause, in `OpsPanel.__init__`, which took `viewer: Any = None`. napari decides
whether to inject a viewer in `_instantiate_dock_widget`
(`napari/_qt/qt_main_window.py`):

```python
for param in sig.parameters.values():
    if param.name == 'napari_viewer': ...
    if param.annotation in ('napari.viewer.Viewer', Viewer, ViewerModel): ...
```

`viewer: Any` matches neither. napari therefore constructed the panel with **no
viewer**, and `_on_done`'s `elif self._viewer is not None:` discarded every
layer output without a word.

Two fixes, and the second matters more than the first:

**1. Name the parameter `napari_viewer`.** The name check is first and is the
only one that can work here: the annotation route compares against the `Viewer`
type or an exact string, and `from __future__ import annotations` turns every
annotation in this module into a string that does not match. There is an NB
comment on the parameter saying so, because this looks like a stylistic choice
and is not.

**2. Never fail quietly.** The no-viewer branch now names the dropped outputs
in a `show_error`:

> *`otsu` produced labels, but this panel has no viewer to put them in…*

A panel with no viewer must not be indistinguishable from an op that does
nothing.

## Why the tests did not catch it

Every existing test constructed `OpsPanel(viewer)` **directly**, passing the
viewer positionally — which works regardless of the parameter's name. The tests
exercised the panel; napari's own instantiation logic was never in the picture.

The regression test goes through napari:

```python
from napari._qt.qt_main_window import _instantiate_dock_widget
widget = _instantiate_dock_widget(OpsPanel, viewer)
assert widget._viewer is not None
```

The general lesson, worth more than the specific fix: **when a framework
constructs your object, test the construction, not just the object.** A
convenient direct constructor is exactly the thing that hides an integration
contract.

## Diagnostics that now exist

The `README` documents three levels; they exist because of the above.

1. `logging.getLogger("napari_skopkit")` at INFO — what ran, with what shapes and
   dtypes, and which layers came back.
2. `NAPARI_SKOPKIT_DEBUG=1` → `Runner(debug=True)` — the worker's own stdout and
   stderr. napari constructs the widget itself, so there is no call site at
   which to pass `debug=True`; an environment variable is the only way in.
3. Run the op in a plain Python session, removing this package entirely.
