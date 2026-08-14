"""Run management and comparison utilities.

Provides:
- Run discovery and metadata loading
- Run comparison and alignment
- Statistical analysis across runs
- Run export/import
- Performance benchmarking
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Run Metadata
# =============================================================================


@dataclass
class RunMetadata:
    """Metadata for a single training run.

    Attributes:
        run_id: Unique identifier for the run.
        run_dir: Path to the run directory.
        name: Human-readable run name.
        description: Run description.
        start_time: Run start timestamp.
        end_time: Run end timestamp.
        duration_seconds: Total training duration.
        max_generations: Maximum generations configured.
        actual_generations: Actual generations completed.
        best_fitness: Best fitness achieved.
        config: Training configuration dictionary.
        tournament_config: Tournament configuration if used.
        tags: List of tags for categorization.
    """
    run_id: str = ""
    run_dir: str = ""
    name: str = "Unnamed Run"
    description: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: float = 0.0
    max_generations: int = 0
    actual_generations: int = 0
    best_fitness: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    tournament_config: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None

    @property
    def progress(self) -> float:
        if self.max_generations == 0:
            return 0.0
        return self.actual_generations / self.max_generations

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "max_generations": self.max_generations,
            "actual_generations": self.actual_generations,
            "best_fitness": self.best_fitness,
            "config": self.config,
            "tournament_config": self.tournament_config,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetadata":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Run Discovery
# =============================================================================


class RunManager:
    """Manages discovery, loading, and comparison of training runs.

    Discovers runs from a runs directory, loads their metadata and metrics,
    and provides comparison utilities.
    """

    def __init__(self, runs_dir: str = "runs"):
        """Initialize the run manager.

        Args:
            runs_dir: Directory containing training runs.
        """
        self.runs_dir = Path(runs_dir)
        self._runs: List[RunMetadata] = []
        self._fitness_data: Dict[str, Dict[str, List[float]]] = {}
        self._tournament_data: Dict[str, List[dict]] = {}
        self._elo_data: Dict[str, Dict[str, List[float]]] = {}

    def discover_runs(self) -> List[RunMetadata]:
        """Discover all training runs in the runs directory.

        Returns:
            List of discovered run metadata.
        """
        self._runs = []

        if not self.runs_dir.exists():
            logger.warning(f"Runs directory not found: {self.runs_dir}")
            return self._runs

        # Look for run directories
        run_patterns = ["run_*", "experiment_*", "gen_*"]
        for pattern in run_patterns:
            for run_path in self.runs_dir.glob(pattern):
                if run_path.is_dir():
                    metadata = self._load_run_metadata(run_path)
                    if metadata:
                        self._runs.append(metadata)

        # Also check for run metadata files
        for meta_file in self.runs_dir.glob("run_*.json"):
            try:
                with open(meta_file) as f:
                    data = json.load(f)
                run_id = data.get("run_id", meta_file.stem)
                run_dir = data.get("run_dir", str(self.runs_dir / run_id))
                metadata = RunMetadata.from_dict(data)
                if metadata not in self._runs:
                    self._runs.append(metadata)
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by start time
        self._runs.sort(key=lambda r: r.start_time or 0, reverse=True)
        return self._runs

    def _load_run_metadata(self, run_dir: Path) -> Optional[RunMetadata]:
        """Load metadata for a single run directory.

        Args:
            run_dir: Path to the run directory.

        Returns:
            RunMetadata or None if loading fails.
        """
        try:
            # Load run.json if exists
            run_json = run_dir / "run.json"
            if run_json.exists():
                with open(run_json) as f:
                    data = json.load(f)
                return RunMetadata.from_dict(data)

            # Load from config or metrics
            config_path = run_dir / "config.yaml"
            metrics_path = run_dir / "metrics.json"

            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                best_fitness = metrics.get("best_fitness", 0.0)
                actual_gens = metrics.get("actual_generations", 0)
            else:
                best_fitness = 0.0
                actual_gens = 0

            # Load config
            config = {}
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}

            return RunMetadata(
                run_id=run_dir.name,
                run_dir=str(run_dir),
                name=run_dir.name,
                best_fitness=best_fitness,
                actual_generations=actual_gens,
                config=config,
            )
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load run metadata from {run_dir}: {e}")
            return None

    def load_fitness_data(self, run_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, List[float]]]:
        """Load fitness data for specified runs.

        Args:
            run_ids: List of run IDs to load. None for all discovered runs.

        Returns:
            Dict mapping run_id to fitness data dict.
        """
        runs_to_load = run_ids or [r.run_id for r in self._runs]
        self._fitness_data = {}

        for run_id in runs_to_load:
            run_dir = self.runs_dir / run_id
            fitness_history_path = run_dir / "fitness_history.json"

            if fitness_history_path.exists():
                try:
                    with open(fitness_history_path) as f:
                        self._fitness_data[run_id] = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load fitness history from {fitness_history_path}")
            else:
                # Try to compute from individual generation metrics
                self._fitness_data[run_id] = self._compute_fitness_from_generations(run_dir)

        return self._fitness_data

    def _compute_fitness_from_generations(self, run_dir: Path) -> Dict[str, List[float]]:
        """Compute fitness data from individual generation metrics.

        Args:
            run_dir: Path to the run directory.

        Returns:
            Fitness data dictionary.
        """
        fitness_data = {"best": [], "mean": [], "median": [], "min": [], "max": [], "std": [], "diversity": []}
        gen_dir = run_dir / "generations"

        if not gen_dir.exists():
            return fitness_data

        for gen_num in range(1000):  # Limit search
            gen_path = gen_dir / f"gen_{gen_num:04d}"
            metrics_path = gen_path / "metrics.json"

            if not metrics_path.exists():
                break

            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                fitness_data["best"].append(metrics.get("best_fitness", 0.0))
                fitness_data["mean"].append(metrics.get("mean_fitness", 0.0))
                fitness_data["median"].append(metrics.get("median_fitness", 0.0))
                fitness_data["min"].append(metrics.get("min_fitness", 0.0))
                fitness_data["max"].append(metrics.get("max_fitness", 0.0))
                fitness_data["std"].append(metrics.get("std_fitness", 0.0))
                fitness_data["diversity"].append(metrics.get("diversity", 0.0))
            except (json.JSONDecodeError, IOError):
                continue

        return fitness_data

    def load_tournament_data(self, run_ids: Optional[List[str]] = None) -> Dict[str, List[dict]]:
        """Load tournament data for specified runs.

        Args:
            run_ids: List of run IDs to load.

        Returns:
            Dict mapping run_id to tournament history.
        """
        runs_to_load = run_ids or [r.run_id for r in self._runs]
        self._tournament_data = {}

        for run_id in runs_to_load:
            run_dir = self.runs_dir / run_id
            history_path = run_dir / "tournament_history.json"

            if history_path.exists():
                try:
                    with open(history_path) as f:
                        self._tournament_data[run_id] = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load tournament history from {history_path}")
            else:
                self._tournament_data[run_id] = []

        return self._tournament_data

    def load_elo_data(self, run_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, List[float]]]:
        """Load ELO history data for specified runs.

        Args:
            run_ids: List of run IDs to load.

        Returns:
            Dict mapping run_id to ELO history.
        """
        runs_to_load = run_ids or [r.run_id for r in self._runs]
        self._elo_data = {}

        for run_id in runs_to_load:
            run_dir = self.runs_dir / run_id
            elo_path = run_dir / "elo_history.json"

            if elo_path.exists():
                try:
                    with open(elo_path) as f:
                        self._elo_data[run_id] = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to load ELO history from {elo_path}")
            else:
                self._elo_data[run_id] = {}

        return self._elo_data

    def compare_runs(self, run_ids: List[str],
                     metric: str = "best",
                     align_by: str = "generation") -> Dict[str, Any]:
        """Compare fitness curves between runs.

        Args:
            run_ids: List of run IDs to compare.
            metric: Which fitness metric to compare ("best", "mean", etc.).
            align_by: How to align runs ("generation" or "progress").

        Returns:
            Comparison data dictionary.
        """
        if not run_ids:
            return {}

        # Load fitness data if not already loaded
        if not self._fitness_data:
            self.load_fitness_data(run_ids)

        comparison = {
            "runs": [],
            "summary": {},
        }

        for run_id in run_ids:
            fitness = self._fitness_data.get(run_id, {})
            values = fitness.get(metric, [])

            run_meta = next((r for r in self._runs if r.run_id == run_id), None)
            name = run_meta.name if run_meta else run_id

            comparison["runs"].append({
                "id": run_id,
                "name": name,
                "values": values,
                "length": len(values),
                "best_value": max(values) if values else 0.0,
                "final_value": values[-1] if values else 0.0,
                "avg_value": float(np.mean(values)) if values else 0.0,
                "std_value": float(np.std(values)) if values else 0.0,
                "improvement": (values[-1] - values[0]) if len(values) > 1 else 0.0,
            })

        # Summary statistics
        if comparison["runs"]:
            all_best = [r["best_value"] for r in comparison["runs"]]
            all_final = [r["final_value"] for r in comparison["runs"]]
            comparison["summary"] = {
                "best_overall": max(all_best),
                "best_run": next(r["id"] for r in comparison["runs"] if r["best_value"] == max(all_best)),
                "worst_overall": min(all_best),
                "avg_improvement": float(np.mean([r["improvement"] for r in comparison["runs"]])),
                "diversity": float(np.std(all_best)),
            }

        return comparison

    def get_run_tags(self) -> Dict[str, List[str]]:
        """Get all unique tags across runs.

        Returns:
            Dict mapping tag to list of run IDs.
        """
        tags: Dict[str, List[str]] = {}
        for run in self._runs:
            for tag in run.tags:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(run.run_id)
        return tags

    def filter_by_tag(self, tag: str) -> List[RunMetadata]:
        """Filter runs by tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of runs with the specified tag.
        """
        return [r for r in self._runs if tag in r.tags]

    def export_comparison(self, run_ids: List[str], output_path: str) -> str:
        """Export run comparison to JSON file.

        Args:
            run_ids: List of run IDs to compare.
            output_path: Output file path.

        Returns:
            Path to the exported file.
        """
        comparison = self.compare_runs(run_ids)
        comparison["exported_at"] = str(Path(output_path).name)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all discovered runs.

        Returns:
            Summary dictionary.
        """
        if not self._runs:
            self.discover_runs()

        return {
            "total_runs": len(self._runs),
            "completed_runs": sum(1 for r in self._runs if r.is_complete),
            "running_runs": sum(1 for r in self._runs if not r.is_complete),
            "avg_best_fitness": float(np.mean([r.best_fitness for r in self._runs])) if self._runs else 0.0,
            "max_best_fitness": max((r.best_fitness for r in self._runs), default=0.0),
            "best_run": max(self._runs, key=lambda r: r.best_fitness).run_id if self._runs else "",
            "total_generations": sum(r.actual_generations for r in self._runs),
            "tags": self.get_run_tags(),
        }
