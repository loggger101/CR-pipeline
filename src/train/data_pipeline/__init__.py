"""Training data pipeline for CR-Pipeline.

Provides:
- Match data collection and storage
- Replay data generation
- Training dataset creation
- Data loading and batching
- Dataset versioning
"""

from .match_collector import (
    MatchCollector,
    MatchData,
    MatchCollectorConfig,
)
from .dataset import (
    TrainingDataset,
    DatasetVersion,
    DatasetSplit,
)

__all__ = [
    "MatchCollector",
    "MatchData",
    "MatchCollectorConfig",
    "TrainingDataset",
    "DatasetVersion",
    "DatasetSplit",
]
