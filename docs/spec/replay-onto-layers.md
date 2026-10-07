# Spec — replaying a recipe onto layers

**Status:** built:
`_replay.py`, `Host.bind`/`Host.replay`, `skop_core.prepare`, parameter forms
(phase 2), axis plans fitted to the bound data (phase 3, as decision 64 in
skop-workflows: re-planned rather than recorded as intent), Run on each
selected layer, Open last export and a recent-recipes list (phase 4). The review questions were
settled with the recommendations: a separate widget, targets drawn by
default, `otsu_2` on a collision, always run, and parameters before axes.

## Goal

A user opens a recipe (a JSON file written by *Export recipe…* or by a
script), chooses which layers stand in for its inputs, and runs it. Its
results arrive as layers, drawn the way the Ops panel draws them. Each run
is recorded in the viewer's session like any other run, so the Record panel
shows the replayed steps, and they can be edited, kept, produced and
exported again.

This is demo step 10 brought into napari. Headless replay and batch already
exist (`skop_core.replay`, `skop_core.batch`); what is missing is a way to
bind a recipe to *layers*, a place to do it, and a run that does not freeze
the viewer.

## What the user does

1. *Plugins → scikit-ops → Replay*, then **Open recipe…**. A recent-recipes
   list and "the recipe the Record panel just exported" come later (phase 4).
2. The panel lists the recipe's **inputs**, one row each:
   - An **array root** gets a layer dropdown. It offers only layers with the
     right number of axes, and only layer types that fit the role the
     recipe's ops give that input (labels for a labels parameter). It is
     pre-filled with the layer whose name matches the root, else the only
     layer that fits, else nothing.
   - A **path root** (decision 60) gets a file picker, showing the recorded
     file name as a hint.
   - Any layer may be chosen, including a result the session already made.
     That chains the recipe onto earlier work, and lineage records it.
3. The panel lists the recipe's **targets** as ticks, defaulting to the
   recipe's own. Any step in the recipe may be ticked (decision 31).
   **Show every step** (off by default) also draws intermediates.
4. **Caution** from the recipe is shown before running: "`labels` was hand
   edited when this was recorded; replay recomputes it" (`edited`,
   `affected`), and kept pixels appear as an extra input to bind (`kept`).
5. **Run** is disabled, with the reason in words, until every needed input
   is bound and every op resolves. Running happens off the GUI thread, with
   one progress bar ("Step 2 of 4: otsu"), the op's own progress inside it,
   and **Cancel**.
6. Results arrive as layers, and the Record panel updates live as each step
   is recorded.

## Architecture

The split already used for recording applies again. Decisions live where
they can be tested without a viewer; Qt only draws and collects clicks.

### skop-workflows (`skop_core.recipe`)

Replay grows from one function into a *prepare, check, run* sequence, so a
host can show what a recipe needs before anything runs:

```python
plan = prepare(recipe, registry, targets=None)   # no data yet
plan.roots      # [RootNeed(name, ndim, dtype, role, path_hint, read_by=[steps])]
plan.steps      # ordered closure of the targets
plan.problems   # ops that do not resolve, as words; empty when runnable
plan.check(bindings)          # UnboundRoot / ndim mismatch, before running
plan.run(bindings, recorder, *, name_step=..., on_step=..., cancelled=...,
         **runner_options) -> Replayed
```

`replay()` and `batch()` stay as thin wrappers over this, so scripts change
nothing. Four changes go with it:

- **A bound array the store already knows is used as it is.** It is not
  declared again under the recipe's root name. In napari every layer is
  already a root (or a step's output) under its own name. Re-declaring it as
  `image` would hijack its identity and move a root other steps depend on.
  The replayed steps then read `root:blobs`, or `smoothed`, which is the
  truth. Only an array the store has never seen is declared under the
  recipe's name.
- **Step names do not collide with the session's.** `name_step(recipe_step,
  op, sources)` is the host's choice. The default follows decision 50:
  replaying onto the same sources is another run of the same step, while
  replaying onto other sources gets a new name (`otsu_2`). `Replayed` maps
  recipe steps to session steps, so the host knows which layer is which.
- **`on_step(index, total, step)`** before each step, and **`cancelled()`**
  polled between steps, with runner options (`on_progress`, `on_start`)
  passed through to each run. Cancelling mid-op is the runner's cancel,
  recorded as `cancelled` (decision 56). Cancelling between steps stops
  before the next one.
- **The role of each root** is read off the ops that consume it (as
  `Host.root_role` does today), so the panel can filter layers by role
  without the recipe having to store roles.

### skop-napari

- `_host.py` (no napari): `Host.replay(plan, bindings, ...)` resolves each
  binding. A layer becomes `array_for(layer)` (the same identity rule as
  running an op, so an edited layer is refused in words). A path stays a
  path. It calls `plan.run` and then `show()` for each step to draw. It is
  tested against `FakeDisplay`.
- `_replay.py`: the **Replay** widget, the fourth contribution in
  `napari.yaml`. It holds the open-recipe button, root rows, target ticks,
  caution text, Run/Cancel and progress. It shares the viewer's `Session`,
  so it uses the same runner, store and Record panel.
- Drawing goes through `ViewerDisplay.show`, which already lays out boxes and
  masks by role.

### Why a separate widget, not a section of the Record panel

The Record panel answers "what happened in this session". Replay is "make
this happen again, here", which needs its own inputs, ticks and progress.
The Record panel is already tall. They share the session, so a replay shows
up in the Record panel's graph as soon as it runs.

## Phases

1. **Minimum replay** (the spec above, without overrides): open a file, bind
   array and path roots, tick targets, run with progress and cancel, draw,
   record. Tests:
   - `skop_core`: `prepare`, `check`, collision naming, and reuse of known
     arrays.
   - `test_host.py`: replay against `FakeDisplay`; refusal of an edited
     layer; chaining onto a step's output.
   - `test_replay.py`: a viewer test with environment-free ops; the Run
     button's reasons; cancel between steps.
2. **Parameters as a form.** Each step expands into the same magicgui form
   the Ops panel builds (`build_inputs`), pre-filled from the recipe. What
   the user changes is passed as `overrides` and recorded as the run's
   parameters. This makes a recipe a reusable tool rather than a fixed
   macro.
3. **Axes that fit the bound layer.** Recipes record concrete axis plans
   today (see your `recipe.json`: a pass-through plan with "256 axis 0" in
   its summary). Replay keeps them when the bound layer has the same number
   of axes. To go further, a recipe records the *intent* (axis labels and the
   per-axis choice: iterate, current position, hand to the op), and replay
   rebuilds the plan against the bound layer with `skop.plan`, shown in the
   same axis-mapping controls the Ops panel uses. A 2-D recipe can then run
   plane by plane over a 3-D stack.
4. **Convenience:** a recent-recipes list, "replay what I just exported"
   without a file, and **Replay on each selected layer**, which is batch over
   layers: one replay per layer, carrying on after a failure (decision 19),
   reported in a table.

## Edge cases the tests should hold

- A recipe naming an op that discovery cannot import: listed in
  `problems`, and Run is disabled.
- A root bound to a layer whose type does not fit the role: not offered.
  If forced from a script, `check` refuses it in words.
- A layer deleted or renamed while bound: the row goes back to unbound.
  Identity is by the host's id, so a rename keeps the binding.
- One layer bound to two roots: allowed. Both roots read the same handle,
  and lineage says so.
- Replaying twice onto the same layers: new runs of the same steps, not new
  steps.
- A failure partway: earlier steps' results are drawn and recorded; the
  failed step is recorded `failed`. A cancel works the same way.
- Recipe format other than 2: refused on open, with the format named.
- Kept pixels in the recipe (`caution.kept`): an extra root to bind, labelled
  "hand-kept pixels for `labels`".

## Questions for review

1. A separate **Replay** widget (recommended) or a section of the Record
   panel?
2. Draw only targets by default (recommended), or every step?
3. On a name collision with a different source, suffix (`otsu_2`,
   recommended), or prefix with the recipe's file name (`recipe:otsu`)?
4. Should replay skip a step that is already current on the same sources
   with the same parameters (Produce-like), or always run (recommended for
   phase 1: always run; it is what "replay" says)?
5. Phase 2 or phase 3 first? I would do parameters first: it is smaller,
   and it is what makes a recipe a tool.
