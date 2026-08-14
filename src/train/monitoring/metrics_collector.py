"""Real-time metrics collection for training monitoring.

Collects and aggregates:
- Fitness metrics (best, mean, std, diversity) per generation
- Training speed (generations/hour, seconds/generation)
- GPU/CPU resource utilization
- Memory usage trends
- Convergence indicators
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """GPU resource statistics."""
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_utilization: float = 0.0
    compute_utilization: float = 0.0
    temperature: float = 0.0
    power_draw_w: float = 0.0
    gpu_index: int = 0


@dataclass
class ResourceStats:
    """System resource statistics."""
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_percent: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0


@dataclass
class TrainingSpeedMetrics:
    """Training speed metrics."""
    current_gen_time_sec: float = 0.0
    avg_gen_time_sec: float = 0.0
    generations_per_hour: float = 0.0
    total_elapsed_sec: float = 0.0
    estimated_time_remaining_sec: float = 0.0


@dataclass
class MonitoringConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    sample_interval_sec: float = 1.0
    gpu_indices: List[int] = field(default_factory=lambda: [0])
    collect_gpu_stats: bool = True
    collect_cpu_stats: bool = True
    collect_io_stats: bool = True
    speed_window: int = 10
    output_dir: str = "runs/monitoring"


class MetricsCollector:
    """Collects and aggregates training metrics in real-time.

    Tracks fitness trends, resource utilization, and training speed.
    Provides convergence detection and bottleneck indicators.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.running = False
        self._start_time: float = 0.0
        self.generation_labels: List[int] = []
        self.best_fitness: List[float] = []
        self.mean_fitness: List[float] = []
        self.std_fitness: List[float] = []
        self.diversity: List[float] = []
        self.gpu_history: List[Dict[str, Any]] = []
        self.cpu_history: List[Dict[str, Any]] = []
        self.gen_times: List[float] = []
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the metrics collector."""
        self.running = True
        self._start_time = time.time()
        logger.info("Metrics collector started")

    def stop(self) -> None:
        """Stop the metrics collector and save data."""
        self.running = False
        self._save_summary()
        logger.info("Metrics collector stopped")

    def record_generation(
        self,
        generation: int,
        fitnesses: List[float],
        diversity: float = 0.0,
    ) -> Dict[str, Any]:
        """Record metrics for a completed generation."""
        if not fitnesses:
            return {}
        best = float(max(fitnesses))
        mean = float(np.mean(fitnesses))
        std = float(np.std(fitnesses))
        self.generation_labels.append(generation)
        self.best_fitness.append(best)
        self.mean_fitness.append(mean)
        self.std_fitness.append(std)
        self.diversity.append(diversity)
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            if len(self.generation_labels) > 1:
                gen_time = elapsed / len(self.generation_labels)
                self.gen_times.append(gen_time)
        speed = self._compute_speed_metrics()
        convergence = self._compute_convergence_indicator()
        metrics = {
            "generation": generation,
            "best_fitness": best,
            "mean_fitness": mean,
            "std_fitness": std,
            "diversity": diversity,
            "speed": speed,
            "convergence": convergence,
            "timestamp": time.time(),
        }
        if generation % 10 == 0 or generation == 1:
            self._save_snapshot(generation)
        return metrics

    def record_gpu_stats(self, gpu_stats: GPUStats) -> None:
        """Record GPU resource stats."""
        if not self.config.collect_gpu_stats:
            return
        self.gpu_history.append({
            "timestamp": time.time(),
            "gpu_index": gpu_stats.gpu_index,
            "memory_used_gb": gpu_stats.memory_used_gb,
            "memory_total_gb": gpu_stats.memory_total_gb,
            "memory_utilization": gpu_stats.memory_utilization,
            "compute_utilization": gpu_stats.compute_utilization,
            "temperature": gpu_stats.temperature,
            "power_draw_w": gpu_stats.power_draw_w,
        })

    def record_cpu_stats(self, cpu_stats: ResourceStats) -> None:
        """Record CPU resource stats."""
        if not self.config.collect_cpu_stats:
            return
        self.cpu_history.append({
            "timestamp": time.time(),
            "cpu_percent": cpu_stats.cpu_percent,
            "memory_used_gb": cpu_stats.memory_used_gb,
            "memory_total_gb": cpu_stats.memory_total_gb,
            "memory_percent": cpu_stats.memory_percent,
        })

    def _compute_speed_metrics(self) -> TrainingSpeedMetrics:
        """Compute training speed metrics."""
        if not self.generation_labels:
            return TrainingSpeedMetrics()
        total_elapsed = time.time() - self._start_time
        num_gens = len(self.generation_labels)
        gen_per_hour = (num_gens / total_elapsed * 3600) if total_elapsed > 0 else 0
        window = min(self.config.speed_window, len(self.gen_times))
        avg_gen_time = (float(np.mean(self.gen_times[-window:])) if self.gen_times else 0)
        return TrainingSpeedMetrics(
            current_gen_time_sec=avg_gen_time,
            avg_gen_time_sec=avg_gen_time,
            generations_per_hour=gen_per_hour,
            total_elapsed_sec=total_elapsed,
            estimated_time_remaining_sec=0.0,
        )

    def _compute_convergence_indicator(self) -> Dict[str, float]:
        """Compute convergence indicators."""
        if len(self.best_fitness) < 10:
            return {"converged": False, "variance": 0.0, "trend": "unknown"}
        window = min(10, len(self.best_fitness))
        recent = self.best_fitness[-window:]
        variance = float(np.var(recent))
        if len(recent) >= 3:
            x = np.arange(len(recent))
            slope = float(np.polyfit(x, recent, 1)[0])
            trend = "improving" if slope > 0.01 else ("declining" if slope < -0.01 else "stable")
        else:
            slope = 0.0
            trend = "insufficient_data"
        return {"converged": variance < 0.01, "variance": variance, "trend": trend, "slope": slope}

    def _save_snapshot(self, generation: int) -> None:
        """Save a metrics snapshot for a generation."""
        snapshot = {
            "generation": generation,
            "generation_labels": self.generation_labels,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.std_fitness,
            "diversity": self.diversity,
            "gen_times": self.gen_times,
            "gpu_history": self.gpu_history[-100:],
            "cpu_history": self.cpu_history[-100:],
            "timestamp": time.time(),
        }
        path = self.output_dir / f"snapshot_gen_{generation:04d}.json"
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

    def _save_summary(self) -> None:
        """Save final monitoring summary."""
        summary = {
            "total_generations": len(self.generation_labels),
            "total_elapsed_sec": time.time() - self._start_time if self._start_time else 0,
            "avg_generation_time": float(np.mean(self.gen_times)) if self.gen_times else 0,
            "final_best_fitness": self.best_fitness[-1] if self.best_fitness else 0,
            "final_mean_fitness": self.mean_fitness[-1] if self.mean_fitness else 0,
            "convergence": self._compute_convergence_indicator(),
        }
        path = self.output_dir / "monitoring_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    def get_fitness_curves(self) -> Dict[str, List[float]]:
        """Get fitness curves as lists."""
        return {
            "best": list(self.best_fitness),
            "mean": list(self.mean_fitness),
            "std": list(self.std_fitness),
            "diversity": list(self.diversity),
            "generation": list(self.generation_labels),
        }

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary."""
        summary = {}
        if self.gpu_history:
            summary["gpu"] = {
                "avg_memory_gb": float(np.mean([g["memory_used_gb"] for g in self.gpu_history])),
                "max_memory_gb": max(g["memory_used_gb"] for g in self.gpu_history),
                "avg_compute": float(np.mean([g["compute_utilization"] for g in self.gpu_history])),
                "avg_temperature": float(np.mean([g["temperature"] for g in self.gpu_history])),
            }
        if self.cpu_history:
            summary["cpu"] = {
                "avg_memory_gb": float(np.mean([c["memory_used_gb"] for c in self.cpu_history])),
                "avg_memory_pct": float(np.mean([c["memory_percent"] for c in self.cpu_history])),
            }
        return summary
