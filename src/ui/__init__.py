"""Desktop UI for driving the CR-Pipeline.

``app`` pulls in Tk and matplotlib, so it is imported lazily: importing this
package on a headless machine (CI, a worker process) must not require a
display.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["launch", "CRPipelineApp"]


def launch(project_root: Optional[str] = None,
           runs_dir: Optional[str] = None) -> None:
    """Open the desktop application window."""
    from .app import launch as _launch
    _launch(project_root, runs_dir)


def __getattr__(name: str):
    if name == "CRPipelineApp":
        from .app import CRPipelineApp
        return CRPipelineApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
