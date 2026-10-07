"""Run scikit-ops ops from napari.

Exposes three widgets. The Ops panel lists every op skop can discover and
runs the chosen one in its own Appose environment -- off the GUI thread, with
progress and cancellation, and with its outputs added to the viewer as the
layer types their roles call for. The Workflows panel is the same panel
showing the other half of the collection. The Record panel runs an op and
writes the call down, so a session can be replayed or exported later. The
Replay panel runs a recipe again on the layers the user chooses.
"""

from __future__ import annotations

from ._panel import OpsPanel, WorkflowsPanel
from ._record import RecordPanel
from ._replay import ReplayPanel

__all__ = ["OpsPanel", "RecordPanel", "ReplayPanel", "WorkflowsPanel"]
