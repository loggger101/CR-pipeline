"""Training dataset management for CR-Pipeline.

Provides:
- Training dataset creation from match data
- Dataset versioning
- Dataset splitting (train/val/test)
- Data loading and batching
- Dataset statistics
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DatasetSplit(Enum):
    """Dataset splits."""
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


@dataclass
class DatasetVersion:
    """Version of a training dataset.

    Attributes:
        dataset_id: Unique dataset identifier.
        version: Version string.
        split: Dataset split.
        match_count: Number of matches.
        state_count: Total state samples.
        created_at: Creation timestamp.
        source_runs: Source run IDs.
        metadata: Additional metadata.
    """
    dataset_id: str
    version: str
    split: DatasetSplit
    match_count: int = 0
    state_count: int = 0
    created_at: float = field(default_factory=time.time)
    source_runs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "split": self.split.name,
            "match_count": self.match_count,
            "state_count": self.state_count,
            "created_at": self.created_at,
            "source_runs": self.source_runs,
            "metadata": self.metadata,
        }


class TrainingDataset:
    """Manages training datasets created from match data.

    Supports:
    - Creating datasets from match collections
    - Splitting into train/val/test sets
    - Loading and batching data
    - Dataset versioning
    """

    def __init__(self, dataset_dir: str = "runs/datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.versions: List[DatasetVersion] = []
        self._load_versions()

    def _load_versions(self) -> None:
        """Load existing dataset versions from disk."""
        versions_path = self.dataset_dir / "versions.json"
        if versions_path.exists():
            with open(versions_path) as f:
                data = json.load(f)
            for v_data in data:
                v_data["split"] = DatasetSplit[v_data["split"]]
                self.versions.append(DatasetVersion(**v_data))

    def _save_versions(self) -> None:
        """Save dataset versions to disk."""
        versions_path = self.dataset_dir / "versions.json"
        with open(versions_path, "w") as f:
            json.dump([v.to_dict() for v in self.versions], f, indent=2)

    def create_dataset(
        self,
        dataset_id: str,
        source_runs: List[str],
        split: DatasetSplit,
        match_filter: Optional[Dict[str, Any]] = None,
    ) -> DatasetVersion:
        """Create a new dataset version from match data.

        Args:
            dataset_id: Unique dataset identifier.
            source_runs: Source run IDs to collect matches from.
            split: Dataset split type.
            match_filter: Optional filter for matches.

        Returns:
            Created DatasetVersion.
        """
        match_count = 0
        state_count = 0
        metadata = {}

        for run_id in source_runs:
            run_dir = Path("runs") / run_id
            if not run_dir.exists():
                continue
            for gen_dir in run_dir.glob("gen_*"):
                summary_files = gen_dir.glob("*_summary.json")
                for summary_path in summary_files:
                    try:
                        with open(summary_path) as f:
                            summary = json.load(f)
                        if match_filter:
                            if not all(summary.get(k) == v for k, v in match_filter.items()):
                                continue
                        match_count += 1
                        state_count += summary.get("state_snapshots_count", 0)
                    except (json.JSONDecodeError, KeyError):
                        continue

        version = DatasetVersion(
            dataset_id=dataset_id,
            version=f"v{len(self.versions) + 1}",
            split=split,
            match_count=match_count,
            state_count=state_count,
            source_runs=source_runs,
            metadata=metadata,
        )
        self.versions.append(version)
        self._save_versions()

        logger.info(f"Created dataset {dataset_id} v{version.version} "
                     f"({match_count} matches, {state_count} states)")
        return version

    def get_dataset_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Get statistics for a dataset.

        Args:
            dataset_id: Dataset identifier.

        Returns:
            Dataset statistics.
        """
        version = next((v for v in self.versions if v.dataset_id == dataset_id), None)
        if not version:
            return {"error": "Dataset not found"}

        return {
            "dataset_id": version.dataset_id,
            "version": version.version,
            "split": version.split.name,
            "match_count": version.match_count,
            "state_count": version.state_count,
            "source_runs": version.source_runs,
            "created_at": version.created_at,
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets."""
        return [v.to_dict() for v in self.versions]
