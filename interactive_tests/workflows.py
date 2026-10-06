"""Launch napari with the panels this repo contributes, on a sample image.

    uv run python interactive_tests/workflows.py

A starting point for a demo rather than a test: it opens a viewer, puts a
synthetic blobs image in it, and docks the scikit-ops panels so a workflow can
be run and recorded by hand. Nothing here is asserted -- ``tests/`` is what
asserts.

The image is made with numpy rather than read from disk, so the script needs
nothing beyond this repo's own environment.
"""

from __future__ import annotations

import napari
import numpy as np


def blobs(shape: tuple[int, int] = (256, 256), count: int = 12, seed: int = 0):
    """A dark image with bright Gaussian blobs, for something to segment."""
    rng = np.random.default_rng(seed)
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    image = np.zeros(shape, dtype=float)
    for _ in range(count):
        y, x = rng.uniform(0, rows), rng.uniform(0, cols)
        sigma = rng.uniform(6.0, 16.0)
        image += rng.uniform(0.4, 1.0) * np.exp(
            -((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2)
        )
    return image + rng.normal(0.0, 0.01, shape)


def main() -> None:
    from skop_napari import OpsPanel, RecordPanel, WorkflowsPanel

    viewer = napari.Viewer()
    viewer.add_image(blobs(), name="image")

    # The whole surface: run a leaf op, run an assembled workflow, and record
    # either into a log that can be replayed or exported.
    viewer.window.add_dock_widget(OpsPanel(viewer), name="Ops", area="right")
    viewer.window.add_dock_widget(
        WorkflowsPanel(viewer), name="Workflows", area="right"
    )
    viewer.window.add_dock_widget(RecordPanel(viewer), name="Record", area="right")

    print(
        "scikit-ops panels docked on the right.\n"
        "  Ops       -- a leaf op, run in its own environment\n"
        "  Workflows -- an assembled workflow; its stages are combos\n"
        "  Record    -- run an op and write the call down, to replay later"
    )
    napari.run()


if __name__ == "__main__":
    main()
