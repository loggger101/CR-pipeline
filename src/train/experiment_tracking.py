"""Experiment tracking for CR-Pipeline.

Provides:
- MLflow-like experiment tracking
- Hyperparameter logging
- Metrics collection and visualization
- Model registry
- Run comparison
- Report generation
- Experiment versioning
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Experiment Data Models
# =============================================================================


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    step: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExperimentRun:
    """Represents a single experiment run.

    Attributes:
        run_id: Unique run identifier.
        experiment_id: Parent experiment ID.
        name: Run name.
        status: Run status (running, completed, failed).
        start_time: Run start timestamp.
        end_time: Run end timestamp.
        params: Hyperparameters.
        metrics: Collected metrics.
        tags: Run tags.
        artifacts: Artifacts produced.
        notes: User notes.
    """
    run_id: str
    experiment_id: str
    name: str = ""
    status: str = "running"
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[MetricPoint]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    notes: str = ""

    @property
    def duration(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def is_active(self) -> bool:
        return self.status == "running"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "params": self.params,
            "metrics": {
                name: [
                    {"value": p.value, "step": p.step, "timestamp": p.timestamp}
                    for p in points
                ]
                for name, points in self.metrics.items()
            },
            "tags": self.tags,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentRun":
        """Create from dictionary."""
        run = cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            name=data.get("name", ""),
            status=data.get("status", "running"),
            start_time=data.get("start_time", time.time()),
            end_time=data.get("end_time"),
            params=data.get("params", {}),
            tags=data.get("tags", []),
            artifacts=data.get("artifacts", []),
            notes=data.get("notes", ""),
        )

        # Reconstruct metrics
        for name, points_data in data.get("metrics", {}).items():
            run.metrics[name] = [
                MetricPoint(
                    name=name,
                    value=p["value"],
                    step=p["step"],
                    timestamp=p.get("timestamp", time.time()),
                )
                for p in points_data
            ]

        return run


@dataclass
class Experiment:
    """Represents an experiment containing multiple runs.

    Attributes:
        experiment_id: Unique experiment ID.
        name: Experiment name.
        description: Experiment description.
        created_at: Creation timestamp.
        runs: List of runs in this experiment.
        tags: Experiment tags.
    """
    experiment_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    runs: List[ExperimentRun] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def add_run(self, run: ExperimentRun) -> None:
        """Add a run to this experiment."""
        self.runs.append(run)

    def get_best_run(self, metric_name: str = "best_fitness") -> Optional[ExperimentRun]:
        """Get the best run based on a metric.

        Args:
            metric_name: Metric to optimize.

        Returns:
            Best run or None.
        """
        if not self.runs:
            return None

        best_run = None
        best_value = -float("inf")

        for run in self.runs:
            if run.status != "completed":
                continue

            points = run.metrics.get(metric_name, [])
            if points:
                current_value = points[-1].value
                if current_value > best_value:
                    best_value = current_value
                    best_run = run

        return best_run

    def get_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics."""
        completed_runs = [r for r in self.runs if r.status == "completed"]

        if not completed_runs:
            return {"total_runs": len(self.runs), "completed_runs": 0}

        # Collect all metrics
        all_metrics = {}
        for run in completed_runs:
            for name, points in run.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                if points:
                    all_metrics[name].append(points[-1].value)

        # Compute statistics
        metric_stats = {}
        for name, values in all_metrics.items():
            if values:
                metric_stats[name] = {
                    "best": max(values),
                    "worst": min(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }

        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "total_runs": len(self.runs),
            "completed_runs": len(completed_runs),
            "active_runs": sum(1 for r in self.runs if r.is_active),
            "failed_runs": sum(1 for r in self.runs if r.status == "failed"),
            "duration_hours": (time.time() - self.created_at) / 3600,
            "metrics": metric_stats,
        }


# =============================================================================
# Experiment Tracker
# =============================================================================


class ExperimentTracker:
    """Tracks experiments and runs for CR-Pipeline.

    Provides:
    - Experiment and run management
    - Metric logging
    - Parameter tracking
    - Artifact storage
    - Run comparison
    - Report generation
    """

    def __init__(self, tracking_dir: str = "experiment_tracking"):
        """Initialize the experiment tracker.

        Args:
            tracking_dir: Directory for tracking data.
        """
        self.tracking_dir = Path(tracking_dir)
        self.tracking_dir.mkdir(parents=True, exist_ok=True)

        self._experiments: Dict[str, Experiment] = {}
        self._active_runs: Dict[str, ExperimentRun] = {}
        self._run_history: List[dict] = []

        # Load existing data
        self._load_existing_data()

    def _load_existing_data(self) -> None:
        """Load existing experiments and runs from disk."""
        exp_dir = self.tracking_dir / "experiments"
        if exp_dir.exists():
            for exp_file in exp_dir.glob("*.json"):
                try:
                    with open(exp_file) as f:
                        data = json.load(f)
                    experiment = Experiment(
                        experiment_id=data["experiment_id"],
                        name=data["name"],
                        description=data.get("description", ""),
                        created_at=data.get("created_at", time.time()),
                        tags=data.get("tags", []),
                    )
                    for run_data in data.get("runs", []):
                        run = ExperimentRun.from_dict(run_data)
                        experiment.add_run(run)
                    self._experiments[experiment.experiment_id] = experiment
                except (json.JSONDecodeError, KeyError):
                    continue

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name.
            description: Experiment description.
            tags: Experiment tags.

        Returns:
            Created Experiment.
        """
        import uuid
        experiment_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tags or [],
        )

        self._experiments[experiment_id] = experiment
        self._save_experiment(experiment)

        logger.info(f"Created experiment: {name} ({experiment_id})")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Experiment or None.
        """
        return self._experiments.get(experiment_id)

    def list_experiments(self, tag: Optional[str] = None) -> List[Experiment]:
        """List experiments, optionally filtered by tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of experiments.
        """
        if tag:
            return [e for e in self._experiments.values() if tag in e.tags]
        return list(self._experiments.values())

    def start_run(
        self,
        experiment_id: str,
        name: str = "",
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ExperimentRun:
        """Start a new run in an experiment.

        Args:
            experiment_id: Parent experiment ID.
            name: Run name.
            params: Hyperparameters to log.
            tags: Run tags.

        Returns:
            Started ExperimentRun.
        """
        import uuid
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            name=name or f"run_{len(experiment.runs) + 1}",
            params=params or {},
            tags=tags or [],
        )

        experiment.add_run(run)
        self._active_runs[run_id] = run
        self._save_experiment(experiment)

        logger.info(f"Started run: {run.name} ({run_id})")
        return run

    def end_run(self, run_id: str, status: str = "completed") -> None:
        """End a run.

        Args:
            run_id: Run ID to end.
            status: Final status.
        """
        run = self._active_runs.get(run_id)
        if not run:
            logger.warning(f"Run {run_id} not found or already ended")
            return

        run.status = status
        run.end_time = time.time()

        # Move to history
        self._run_history.append(run.to_dict())
        del self._active_runs[run_id]

        # Update experiment
        experiment = self._experiments.get(run.experiment_id)
        if experiment:
            self._save_experiment(experiment)

        logger.info(f"Ended run: {run.name} ({run_id}) - {status}")

    def log_metric(self, run_id: str, name: str, value: float, step: int = 0) -> None:
        """Log a metric for a run.

        Args:
            run_id: Run ID.
            name: Metric name.
            value: Metric value.
            step: Step/iteration number.
        """
        run = self._active_runs.get(run_id)
        if not run:
            logger.warning(f"Run {run_id} not active, metric not logged")
            return

        if name not in run.metrics:
            run.metrics[name] = []

        run.metrics[name].append(MetricPoint(name=name, value=value, step=step))

    def log_metrics_batch(self, run_id: str, metrics: Dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once.

        Args:
            run_id: Run ID.
            metrics: Dictionary of metric name to value.
            step: Step/iteration number.
        """
        for name, value in metrics.items():
            self.log_metric(run_id, name, value, step)

    def log_param(self, run_id: str, name: str, value: Any) -> None:
        """Log a parameter for a run.

        Args:
            run_id: Run ID.
            name: Parameter name.
            value: Parameter value.
        """
        run = self._active_runs.get(run_id)
        if run:
            run.params[name] = value

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log multiple parameters.

        Args:
            run_id: Run ID.
            params: Dictionary of parameter name to value.
        """
        for name, value in params.items():
            self.log_param(run_id, name, value)

    def add_artifact(self, run_id: str, artifact_path: str) -> None:
        """Add an artifact to a run.

        Args:
            run_id: Run ID.
            artifact_path: Path to the artifact.
        """
        run = self._active_runs.get(run_id)
        if run:
            run.artifacts.append(artifact_path)

    def add_tag(self, run_id: str, tag: str) -> None:
        """Add a tag to a run.

        Args:
            run_id: Run ID.
            tag: Tag to add.
        """
        run = self._active_runs.get(run_id)
        if run and tag not in run.tags:
            run.tags.append(tag)

    def set_notes(self, run_id: str, notes: str) -> None:
        """Set notes for a run.

        Args:
            run_id: Run ID.
            notes: Notes text.
        """
        run = self._active_runs.get(run_id)
        if run:
            run.notes = notes

    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get a run by ID.

        Args:
            run_id: Run ID.

        Returns:
            ExperimentRun or None.
        """
        return self._active_runs.get(run_id) or next(
            (r for exp in self._experiments.values() for r in exp.runs if r.run_id == run_id),
            None,
        )

    def compare_runs(
        self,
        run_ids: List[str],
        metric_name: str = "best_fitness",
    ) -> Dict[str, Any]:
        """Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare.
            metric_name: Metric to compare on.

        Returns:
            Comparison results.
        """
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        if not runs:
            return {}

        # Collect metric values
        metric_values = {}
        for run in runs:
            points = run.metrics.get(metric_name, [])
            if points:
                metric_values[run.run_id] = {
                    "best": max(p.value for p in points),
                    "final": points[-1].value,
                    "mean": float(np.mean([p.value for p in points])),
                    "steps": len(points),
                }

        # Compute statistics
        all_best = [v["best"] for v in metric_values.values()]

        return {
            "metric": metric_name,
            "runs": metric_values,
            "best_run": max(metric_values.keys(), key=lambda k: metric_values[k]["best"]) if all_best else None,
            "best_value": max(all_best) if all_best else None,
            "avg_best": float(np.mean(all_best)) if all_best else None,
            "spread": max(all_best) - min(all_best) if all_best else 0,
        }

    def generate_report(
        self,
        experiment_id: str,
        metric_name: str = "best_fitness",
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a text report for an experiment.

        Args:
            experiment_id: Experiment ID.
            metric_name: Primary metric.
            output_path: Output file path.

        Returns:
            Report text.
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        summary = experiment.get_summary()
        best_run = experiment.get_best_run(metric_name)

        report_lines = [
            "=" * 60,
            f"EXPERIMENT REPORT: {experiment.name}",
            "=" * 60,
            f"Experiment ID: {experiment.experiment_id}",
            f"Description: {experiment.description}",
            f"Created: {datetime.fromtimestamp(experiment.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {summary['duration_hours']:.2f} hours",
            f"Tags: {', '.join(experiment.tags)}",
            "",
            "-" * 40,
            "SUMMARY",
            "-" * 40,
            f"Total Runs: {summary['total_runs']}",
            f"Completed: {summary['completed_runs']}",
            f"Active: {summary['active_runs']}",
            f"Failed: {summary['failed_runs']}",
            "",
        ]

        if best_run:
            report_lines.extend([
                "-" * 40,
                "BEST RUN",
                "-" * 40,
                f"Run ID: {best_run.run_id}",
                f"Name: {best_run.name}",
                f"Status: {best_run.status}",
                f"Duration: {best_run.duration:.1f}s",
                f"Best {metric_name}: {summary['metrics'].get(metric_name, {}).get('best', 'N/A')}",
                "",
            ])

        # Best run parameters
        if best_run:
            report_lines.extend([
                "-" * 40,
                "BEST RUN PARAMETERS",
                "-" * 40,
            ])
            for name, value in sorted(best_run.params.items()):
                report_lines.append(f"  {name}: {value}")
            report_lines.append("")

        # Metric statistics
        if "metrics" in summary:
            report_lines.extend([
                "-" * 40,
                f"METRIC STATISTICS ({metric_name})",
                "-" * 40,
            ])
            stats = summary["metrics"].get(metric_name, {})
            if stats:
                report_lines.extend([
                    f"  Best:  {stats.get('best', 'N/A')}",
                    f"  Worst: {stats.get('worst', 'N/A')}",
                    f"  Mean:  {stats.get('mean', 'N/A'):.4f}",
                    f"  Std:   {stats.get('std', 'N/A'):.4f}",
                    f"  Count: {stats.get('count', 0)}",
                ])
            report_lines.append("")

        # Top runs
        completed_runs = [r for r in experiment.runs if r.status == "completed"]
        if completed_runs:
            ranked = sorted(
                completed_runs,
                key=lambda r: max((p.value for p in r.metrics.get(metric_name, [])), default=0),
                reverse=True,
            )[:10]

            report_lines.extend([
                "-" * 40,
                "TOP 10 RUNS",
                "-" * 40,
                f"{'Rank':<6}{'Run ID':<35}{'Score':>10}{'Duration':>10}",
                "-" * 61,
            ])

            for i, run in enumerate(ranked, 1):
                points = run.metrics.get(metric_name, [])
                score = points[-1].value if points else 0
                report_lines.append(
                    f"{i:<6}{run.run_id[:32]:<35}{score:>10.4f}{run.duration:>10.1f}s"
                )
            report_lines.append("")

        report = "\n".join(report_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to disk."""
        exp_dir = self.tracking_dir / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        filepath = exp_dir / f"{experiment.experiment_id}.json"
        data = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "created_at": experiment.created_at,
            "tags": experiment.tags,
            "runs": [r.to_dict() for r in experiment.runs],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_active_runs(self) -> List[ExperimentRun]:
        """Get all active runs."""
        return list(self._active_runs.values())

    def get_experiment_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all experiments."""
        return [exp.get_summary() for exp in self._experiments.values()]
