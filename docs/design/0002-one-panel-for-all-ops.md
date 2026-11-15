# 0002 — One Ops widget, not one per op

## The tension

The natural napari plugin shape is one contribution per command: each op would
appear in the *Plugins* menu under its own name, get its own dock widget, and
be findable by search. That is what a user expects and what every mature napari
plugin does.

It is incompatible with how skop finds its ops. npe2 wants a **static
manifest** — `napari.yaml`, read without importing anything — while
`skop.discover()` finds ops by **importing** every module under `skop.ops` and
reading their signatures. A contribution per op means generating `napari.yaml`
at build time from a discovery run, which costs the property that makes this
collection pleasant to work on: drop a new op file in and it appears.

## The decision

One contribution, `napari_skopkit._panel:OpsPanel`, carrying an op picker.
Choosing an op rebuilds the input widgets below it. `napari.yaml` has one entry
and never changes.

```
op:      [ segment: stardist2d  ▾ ]
         Segment nuclei with StarDist 2D.
image:   [ blobs ▾ ]
prob:    [====|=====] 0.5
         Environment: stardist-tf
         [ Run ]
```

Prior art: [napari-imagej](https://github.com/imagej/napari-imagej) exposes all
of Fiji's SciJava modules the same way and for the same reason — an unbounded,
dynamically-discovered command set behind one widget with a search field.

## What it costs

- Ops are not in the *Plugins* menu, not in napari's command search, and cannot
  be bound to a shortcut individually.
- Layer-type filtering is per-widget rather than per-command, so an op that
  needs labels does not gray itself out when no labels layer exists.
- One widget means one selection state; two ops cannot be open side by side.

All three are real, all three are consequences of the same missing feature, and
none is worth a build step.

## Why not generate the manifest

Considered seriously. Rejected on the workflow: it puts a code-generation step
between writing an op and running it, and that step has to run in an
environment able to import every op module — which is the host environment, so
it *would* work, but it means a stale manifest is now a failure mode. The
collection is meant to be extended by dropping in a file. It should not also
require remembering to regenerate something.

A weaker version — generate the manifest in CI, commit it — has the same
failure mode plus merge conflicts in a generated file.

## The exit

The napari team has plans to make dynamic registration of commands feasible.
When that lands, `_panel.py` is what gets split up: the per-op form building is
already in `_widget.py` and already independent of the picker. See
[spec/dynamic-registration.md](../spec/dynamic-registration.md).

This is written at the top of `_panel.py` as well, because that file's module
docstring is where someone will actually be standing when they wonder why.
