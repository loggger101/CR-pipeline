"""Tournament data pipeline for CR-Pipeline.

Provides:
- Tournament result collection and storage
- ELO rating history tracking
- Head-to-head record aggregation
- Tournament dataset creation
- Bracket visualization data
"""

from .tournament_collector import (
    TournamentCollector,
    TournamentMatchData,
    TournamentCollectorConfig,
    TournamentDataset,
    TournamentDatasetVersion,
)

__all__ = [
    "TournamentCollector",
    "TournamentMatchData",
    "TournamentCollectorConfig",
    "TournamentDataset",
    "TournamentDatasetVersion",
]
