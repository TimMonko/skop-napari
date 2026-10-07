# skop-napari

Run [scikit-ops](https://github.com/apposed/scikit-ops) ops from napari.

Adds three dock widgets under *Plugins → scikit-ops*:

- **Ops** lists every op skop can discover, builds a form for the chosen one,
  and runs it in its own Appose environment — off the GUI thread, with
  progress and cancellation, and with its outputs added to the viewer as the
  layer types their roles call for.
- **Workflows** is the same panel showing the ops that call other ops.
- **Record** writes down what the other two run, through
  skop-workflows: the inputs,
  a graph of what was made from what, which steps are stale, edited or kept,
  *Produce* to bring ticked steps up to date, and *Export recipe…* to write
  a recipe a script can replay. It runs nothing itself.

```sh
uv sync --all-groups
uv run napari
```

## What it does with an op

An op is an ordinary Python function carrying enough annotation to describe
itself. This package reads that description and nothing else — it knows about
no op in particular.

| From the op | Becomes |
| --- | --- |
| a parameter's type | a magicgui widget |
| `Annotated[..., {...}]` UI hints | that widget's type, range, step |
| the docstring's `Args:` section | the widget's tooltip |
| an `ImageData`/`LabelsData`/… role | a layer combo box, not a file picker |
| an `Axes(...)` declaration | an axes field, a mapping per slot, a choice per leftover axis |
| an `Out[...]` parameter | nothing — a user is never asked for a buffer |
| the op's `env` | the Appose environment it runs in |
| `skop.progress(...)` | the progress bar |
| `skop.cancel_requested()` | the Cancel button |
| an output's role | an Image / Labels / Points / … layer |
| an output with no role | a row in the results panel |
| a failure | a napari error notification, with traceback |

That last pair is the point of skop's roles. `otsu` returns `LabelsData`, so
its result arrives as a Labels layer rather than a grayscale image; `unseg`
returns two `LabelsData` and two `int`s, so it produces two Labels layers and
two rows of counts.

## Layout

```
src/skop_napari/_roles.py    Role -> napari type, and laying boxes/masks out
src/skop_napari/_axes.py     working out what a layer's axes are (no Qt)
src/skop_napari/_plans.py    the axes field, slot mapping and axis dispositions
src/skop_napari/_widget.py   OpSpec -> magicgui widgets (imports no napari)
src/skop_napari/_choices.py  a workflow's stages: a combo per chosen op
src/skop_napari/_run.py      running an op off the GUI thread
src/skop_napari/_panel.py    the Ops and Workflows widgets
src/skop_napari/_host.py     recording bookkeeping: layers <-> runs (no napari)
src/skop_napari/_session.py  one recorder, store and runner per viewer
src/skop_napari/_graph.py    the Record panel's graph canvas
src/skop_napari/_record.py   the Record widget
```

`_widget.py` deliberately imports nothing from napari: the one napari-shaped
decision, what type each parameter should be rendered as, is passed in as
`annotation_for`. If a second magicgui host ever turns up, that module is what
moves out.

## Design notes

The long form, with the alternatives considered and the evidence behind each
call, is in [`docs/`](docs/README.md). In brief:

**One widget, not one per op.** npe2 wants a static manifest, while skop finds
its ops by importing them, so a contribution per op would mean generating the
manifest at build time and losing the ability to drop in a new op and see it
appear. The napari team is working on dynamic registration; `_panel.py` is
what gets split up when it lands.

**Widgets are built one parameter at a time**, rather than by handing magicgui
a synthesized function. Real ops contain parameters magicgui cannot render —
`unseg` has `tuple[int, ...]` — and one of those must not take the whole panel
down. An unrenderable parameter with a default is quietly left at its default
and reported in the notes; one without a default disables the Run button and
says why.

**Roles are guessed here, never in skop.** skop reports `role is None` for an
unannotated array rather than assuming it is an image. This package makes that
assumption, because it has a viewer to make it for.

**So are axes.** An op declaring `Axes("y", "x", "c?")` consumes two or three
axes, so handing it a `(z, y, x)` stack means deciding something: which axes
feed the op, and what becomes of the rest. skop supplies a default answer and
flags where it looks doubtful, but forbids nothing — whether a stack should be
processed plane by plane or whole is a property of the experiment, not of the
op. This package works out what the layer's axes are in the first place — from
its metadata, an xarray's dims, NGFF axes, or napari's own `axis_labels` and
`rgb` — and leaves unnamed whatever none of those say, which napari calls `-3`
`-2` `-1` and so does the panel. The mapping is then seeded from the axes the
viewer is displaying, whatever they are called, and edited: a combo per slot,
and iterate/current-position/hand-to-the-op per leftover axis. Real names are
written back onto the layer and stamped onto results, so a session gets more
certain as it goes.

**Errors go to napari, not into the panel.** An op fails in another process,
running another interpreter, so the interesting part of the failure is that
process's traceback. `notification_manager.receive_error` carries it, keeps it
in the notification history, and looks like every other napari error — none of
which a one-line field in the widget could do.

**Environment building reports through Appose's builder callbacks.**
`Runner.subscribe_build_progress/_output/_error` feed the same progress bar
the op itself uses, since building is by far the slowest part of a first run
and happens inside `run()` with nothing else to show for it.

A fresh build drives the bar like this:

```
Preparing environment: probe          (indeterminate)
Installing conda packages             0/30
Installing conda packages             30/30
Done                                  1/1
✔ The default environment has been installed.
```

Two things worth knowing about those channels:

- **Subscribing to progress is what enables it.** Appose's
  `PixiInstallMonitor` only wires up when a progress subscriber exists, and
  it works by running pixi under `-vv` and reading the phase transitions out
  of its log. Appose 0.11 has no monitor at all and reports only a final
  summary line, which is why both projects source Appose from its main
  branch for now.
- **`subscribe_error` is the stderr stream, not a failure report.** Pixi
  writes ordinary status there, success message included — and under the
  `-vv` the monitor injects, its entire debug log too. So this channel is
  logged in full and filtered before it reaches the progress bar: lines that
  look like log records (`DEBUG pixi_config: ...`) are noise, and only
  human-facing lines are shown. A build that genuinely fails raises out of
  `run()` and takes the normal error path.

The `Preparing environment: <id>` label covers the gap before pixi says
anything, which on a cold cache can be a while.

## When an op seems to do nothing

Ops run in another process, so a failure there is easy to lose. Three
increasingly loud ways to see what is going on, all from a terminal-launched
napari (`uv run napari`) or its built-in console:

**1. The panel's own log** — what it ran, with what, and what came back:

```python
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("skop_napari").setLevel(logging.INFO)
```

**2. The worker's stdout and stderr** — what the op is doing inside its own
process, including anything it prints:

```sh
SKOP_NAPARI_DEBUG=1 uv run napari
```

napari builds this widget itself, so there is no call site at which to pass
`Runner(debug=True)`; the environment variable is the way in.

**3. Run the op without a GUI at all**, which removes every layer of this
package from the picture:

```python
import skop
from skop.ops.threshold import otsu
with skop.Runner(debug=True) as runner:
    print(runner.run(otsu, image=my_array))
```

Errors from an op become napari error notifications, carrying the worker
process's traceback — check the notification button in the status bar, since
a notification that has come and gone is still in that list.

## Development

```sh
uv sync --all-groups
uv run pytest
uv run ruff check --fix && uv run ruff format
```

The tests drive a real napari viewer and run real ops in the `minimal` and
`skimage` Appose environments, which are built on first use.

Note that they need a real Qt platform: `QT_QPA_PLATFORM=offscreen` has no GL
context, and napari's vispy canvas segfaults on startup without one.
