"""CR-Pipeline: Visualization.

Provides:
- Training dashboard (Streamlit)
- Fitness metrics computation
- Replay viewer for simulation matches
- Live game visualization overlay
"""

from .metrics import TrainingMetrics
from .dashboard import run_dashboard, render_fitness_curves
from .replay import ReplayViewer, ReplayFrame
from .live_game_view import (
    LiveGameView,
    LiveOverlayData,
    ScreenCapture,
)

__all__ = [
    "TrainingMetrics",
    "run_dashboard",
    "render_fitness_curves",
    "ReplayViewer",
    "ReplayFrame",
    "LiveGameView",
    "LiveOverlayData",
    "ScreenCapture",
]
