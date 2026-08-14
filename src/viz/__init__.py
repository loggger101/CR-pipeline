"""CR-Pipeline: Visualization.

Provides:
- Training dashboard (Streamlit)
- Fitness metrics computation with advanced statistics
- Run management and comparison
- Tournament visualization (brackets, ELO, H2H charts)
- Replay viewer for simulation matches
- Live game visualization overlay
- Simulation rendering for arena visualization
"""

from .metrics import TrainingMetrics
from .runs_manager import RunManager, RunMetadata
from .dashboard import run_dashboard, render_fitness_curves
from .tournament_viz import (
    TournamentSummary,
    render_bracket_ascii,
    create_elo_progression_chart,
    create_win_rate_chart,
    create_h2h_matrix_chart,
    compute_tournament_summary,
    render_tournament_dashboard,
    render_elo_history_dashboard,
    tournament_result_to_dict,
    tournament_result_from_dict,
)
from .replay import ReplayViewer, ReplayFrame
from .live_game_view import LiveGameView, LiveGameOverlay
from .rendering import SimulationRenderer
from .reports import (
    ReportGenerator,
    ReportType,
)

__all__ = [
    "TrainingMetrics",
    "RunManager",
    "RunMetadata",
    "run_dashboard",
    "render_fitness_curves",
    "TournamentSummary",
    "render_bracket_ascii",
    "create_elo_progression_chart",
    "create_win_rate_chart",
    "create_h2h_matrix_chart",
    "compute_tournament_summary",
    "render_tournament_dashboard",
    "render_elo_history_dashboard",
    "tournament_result_to_dict",
    "tournament_result_from_dict",
    "ReportGenerator",
    "ReportType",
    "ReplayViewer",
    "ReplayFrame",
    "LiveGameView",
    "LiveGameOverlay",
    "SimulationRenderer",
]
