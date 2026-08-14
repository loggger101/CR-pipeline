"""Bottleneck detection for training pipeline.

Detects:
- CPU bottleneck (low GPU utilization)
- GPU bottleneck (high GPU utilization, slow generations)
- I/O bottleneck (slow checkpoint save/load)
- Memory bottleneck (OOM risk)
- Data pipeline bottleneck (slow data loading)
- Communication bottleneck (slow worker sync)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """Types of bottlenecks."""
    CPU_BOUND = auto()
    GPU_BOUND = auto()
    IO_BOUND = auto()
    MEMORY_BOUND = auto()
    DATA_PIPELINE = auto()
    COMMUNICATION = auto()
    NONE = auto()


@dataclass
class BottleneckReport:
    """Report of detected bottlenecks."""
    bottleneck_type: BottleneckType
    severity: str = "low"
    description: str = ""
    recommendation: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


class BottleneckDetector:
    """Detects training bottlenecks from resource metrics."""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def detect(
        self,
        gpu_compute_pct: float = 0.0,
        gpu_memory_pct: float = 0.0,
        cpu_percent: float = 0.0,
        gen_time_sec: float = 0.0,
        checkpoint_time_sec: float = 0.0,
        data_load_time_sec: float = 0.0,
        worker_sync_time_sec: float = 0.0,
    ) -> List[BottleneckReport]:
        """Detect bottlenecks from current metrics."""
        reports = []

        if gpu_compute_pct > 95 and gen_time_sec > 60:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.GPU_BOUND,
                severity="high" if gpu_compute_pct > 98 else "medium",
                description=f"GPU compute saturated at {gpu_compute_pct:.1f}%",
                recommendation="Consider reducing batch size or using gradient accumulation",
                metrics={"gpu_compute": gpu_compute_pct, "gen_time": gen_time_sec},
            ))

        if cpu_percent > 90 and gpu_compute_pct < 50:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.CPU_BOUND,
                severity="high",
                description=f"CPU saturated at {cpu_percent:.1f}% while GPU idle at {gpu_compute_pct:.1f}%",
                recommendation="Increase num_workers or optimize data preprocessing",
                metrics={"cpu_percent": cpu_percent, "gpu_compute": gpu_compute_pct},
            ))

        if gpu_memory_pct > 90:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                severity="critical" if gpu_memory_pct > 95 else "high",
                description=f"GPU memory at {gpu_memory_pct:.1f}%",
                recommendation="Reduce population size or batch size",
                metrics={"gpu_memory": gpu_memory_pct},
            ))

        if checkpoint_time_sec > 30:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.IO_BOUND,
                severity="medium",
                description=f"Checkpoint save taking {checkpoint_time_sec:.1f}s",
                recommendation="Use faster storage or reduce checkpoint frequency",
                metrics={"checkpoint_time": checkpoint_time_sec},
            ))

        if data_load_time_sec > gen_time_sec * 0.3 and gen_time_sec > 0:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.DATA_PIPELINE,
                severity="medium",
                description=f"Data loading takes {data_load_time_sec:.1f}s ({data_load_time_sec/gen_time_sec*100:.0f}% of gen time)",
                recommendation="Pre-compute features or use faster data loading",
                metrics={"data_load_time": data_load_time_sec, "gen_time": gen_time_sec},
            ))

        if worker_sync_time_sec > gen_time_sec * 0.2 and gen_time_sec > 0:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.COMMUNICATION,
                severity="low",
                description=f"Worker sync takes {worker_sync_time_sec:.1f}s ({worker_sync_time_sec/gen_time_sec*100:.0f}% of gen time)",
                recommendation="Reduce worker count or use async communication",
                metrics={"sync_time": worker_sync_time_sec, "gen_time": gen_time_sec},
            ))

        if not reports:
            reports.append(BottleneckReport(
                bottleneck_type=BottleneckType.NONE,
                severity="low",
                description="No significant bottlenecks detected",
                recommendation="",
                metrics={},
            ))

        self.history.append({
            "gpu_compute": gpu_compute_pct,
            "gpu_memory": gpu_memory_pct,
            "cpu": cpu_percent,
            "gen_time": gen_time_sec,
            "bottleneck": reports[0].bottleneck_type.name,
            "severity": reports[0].severity,
        })

        return reports

    def get_trend(self, window: int = 10) -> Dict[str, Any]:
        """Get bottleneck trend over recent history."""
        if len(self.history) < 2:
            return {"trend": "insufficient_data"}
        recent = self.history[-window:]
        bottleneck_counts = {}
        for entry in recent:
            bn = entry["bottleneck"]
            bottleneck_counts[bn] = bottleneck_counts.get(bn, 0) + 1
        dominant = max(bottleneck_counts, key=bottleneck_counts.get)
        dominant_pct = bottleneck_counts[dominant] / len(recent) * 100
        return {
            "dominant_bottleneck": dominant,
            "dominant_frequency_pct": dominant_pct,
            "avg_gen_time": float(np.mean([e["gen_time"] for e in recent])),
            "avg_gpu_compute": float(np.mean([e["gpu_compute"] for e in recent])),
            "avg_cpu": float(np.mean([e["cpu"] for e in recent])),
        }
