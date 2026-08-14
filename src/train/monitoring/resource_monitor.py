"""Resource monitoring for GPU and CPU.

Provides:
- GPU resource monitoring via nvidia-smi
- CPU resource monitoring via psutil
- Memory usage tracking
- Disk I/O monitoring
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


class GPUResourceMonitor:
    """GPU resource monitor using nvidia-smi."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index

    def get_stats(self) -> GPUStats:
        """Get current GPU statistics."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                values = [float(v.strip()) for v in result.stdout.split(",")]
                return GPUStats(
                    memory_used_gb=values[0] / 1024 if len(values) > 0 else 0,
                    memory_total_gb=values[1] / 1024 if len(values) > 1 else 0,
                    compute_utilization=values[2] if len(values) > 2 else 0,
                    memory_utilization=values[3] if len(values) > 3 else 0,
                    temperature=values[4] if len(values) > 4 else 0,
                    power_draw_w=values[5] if len(values) > 5 else 0,
                    gpu_index=self.gpu_index,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return GPUStats(gpu_index=self.gpu_index)


class CPUResourceMonitor:
    """CPU resource monitor using psutil."""

    def get_stats(self) -> ResourceStats:
        """Get current CPU and memory statistics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            memory = psutil.virtual_memory()
            return ResourceStats(
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                memory_used_gb=memory.used / (1024 ** 3),
                memory_total_gb=memory.total / (1024 ** 3),
                memory_percent=memory.percent,
            )
        except ImportError:
            logger.warning("psutil not installed, returning default stats")
            return ResourceStats()
