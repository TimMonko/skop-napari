# 0001 — No `skop-magicgui` distribution

## The question

magicgui is not napari. It renders widgets for any Qt application, and the bulk
of what this package does — turn an `OpSpec` into a form — needs magicgui and
nothing else. So: should there be a `skop-magicgui` package underneath
`napari-skopkit`, for hosts that want op forms without a viewer?

The instinct to say yes is strong, because the layering is real. It was still
the wrong call.

## The decision

**No third distribution.** Instead:

- The genuinely shared thing — the *vocabulary* — went into skop core, as
  [semantic roles](https://github.com/apposed/scikit-ops/blob/main/docs/design/0003-semantic-roles.md).
  That is what a second front end actually needs, and it needs it whether or
  not that front end uses magicgui at all. A Fiji front end reads `Role` and
  has never heard of magicgui.
- The magicgui-but-not-napari code lives in `src/napari_skopkit/_widget.py`, a
  module that **imports nothing from napari** and is enforced to stay that way
  by the one seam described below.

So the separation exists at module granularity, where it costs nothing, rather
than at package granularity, where it costs a repository, a release cadence, a
version matrix and a dependency to keep in step.

## The seam

`_widget.py` needs exactly one napari-shaped decision: what type should each
parameter be *rendered* as? A role-annotated image parameter should become a
layer combo box in napari, and something else entirely elsewhere. So that
decision is passed in:

```python
def build_inputs(spec: OpSpec, annotation_for: Callable[[ParamSpec], Any]) -> Inputs:
```

`_roles.py` supplies `annotation_for`, and it is the only napari-specific part
of the widget-building path — two lookup tables and a fallback:

```python
_INPUT_TYPES: dict[Role, Any] = {Role.image: nt.ImageData, ...}
_LAYER_TYPES: dict[Role, str] = {Role.image: "image", ...}

def annotation_for(param):
    if param.role is not None:
        return _INPUT_TYPES[param.role]
    if param.type is np.ndarray:
        return nt.ImageData      # the guess skop refuses to make
    return param.type
```

That `np.ndarray → ImageData` line is the front end taking the guess skop
declines to. It is correct *here* because there is a viewer to make the
assumption for; it would be wrong in skop, where nothing downstream could tell
the guess from a declaration.

## Why the seam and not the package

The value of a package split is that it *forces* the boundary to stay clean.
The value of a module split is that the boundary is clean as long as anyone is
looking. That is a real difference, and the trade was made on evidence:

- **There is no second host.** Not "not yet planned" — not one. The next front
  end on the roadmap is Fiji, which is Java and shares no widget code with
  this at all. A `skop-magicgui` package would have exactly one consumer for
  the foreseeable future.
- **The split is cheap to perform later and expensive to maintain early.**
  `_widget.py` already takes its one napari dependency as an argument. Making
  it a package is: move the file, add a `pyproject.toml`, add a dependency.
  Making it a package *now* means maintaining that release process before
  anything needs it.
- **A premature package would have absorbed the wrong thing.** The first
  sketch of `skop-magicgui` had roles in it, since roles are what the widget
  layer reads. That would have put the vocabulary a Fiji front end needs
  behind a magicgui dependency. Working out that roles belong in skop core is
  what dissolved the case for the package.

## The promotion criterion

Promote `_widget.py` to its own distribution when a second magicgui-using host
appears and needs it. Until then, the README documents the file as the thing
that moves, and the `annotation_for` parameter keeps the move mechanical.
