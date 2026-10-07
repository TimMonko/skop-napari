# skop-napari documentation

Same convention as [scikit-ops](https://github.com/apposed/scikit-ops):

```
docs/design/NNNN-*.md   Settled. Why the code is shaped this way, and what
                        else was considered. Written after the fact.
docs/spec/*.md          Proposed. Not built. Deleted or graduated into a
                        design doc once it is.
```

The [README](../README.md) is the short form. These are for someone who needs
to change the panel and wants to know which of its oddities are load-bearing.

## Design

| | |
| --- | --- |
| [0001](design/0001-no-magicgui-package.md) | Why there is no `skop-magicgui` distribution |
| [0002](design/0002-one-panel-for-all-ops.md) | One Ops widget, not one contribution per op |
| [0003](design/0003-building-input-widgets.md) | One parameter at a time, and surviving the ones that fail |
| [0004](design/0004-running-off-the-gui-thread.md) | Threads, cancellation, and outliving the widget |
| [0005](design/0005-reporting-through-napari.md) | Errors, build progress, and the bug that ate every output |
| [0006](design/0006-axis-awareness.md) | Guessing a layer's axes (adaptation UI superseded by 0007, rung four by 0008) |
| [0007](design/0007-axis-mapping-ui.md) | Editing the axis mapping, and warning instead of forbidding |
| [0008](design/0008-naming-less-and-looking-more.md) | Leaving axes unnamed, and mapping from what the viewer displays |

## Spec

| | |
| --- | --- |
| [dynamic-registration.md](spec/dynamic-registration.md) | Splitting the panel up when npe2 allows it |
| [replay-onto-layers.md](spec/replay-onto-layers.md) | Replaying a recipe onto layers: binding inputs, running, drawing |
