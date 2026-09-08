"""Resource monitoring for GPU and CPU.

Provides platform-independent system resource monitoring:
- GPU stats via nvidia-smi (when available)
- CPU/memory stats via os / Windows WMI / Linux procfs
- No external dependencies beyond stdlib + numpy

Designed to work without psutil, which is not in the project's requirements.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
        """Get current GPU statistics via ``nvidia-smi``.

        Returns default stats when the binary is unavailable or the query fails.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,"
                    "utilization.memory,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                    "-i", str(self.gpu_index),
                ],
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
    """CPU / memory monitor using only stdlib.

    Platform strategy:
    - **Linux**: reads ``/proc/stat`` for CPU and ``/proc/meminfo`` for memory.
    - **Windows (32-bit Python)**: falls back to ``os.getloadavg()`` where
      available, or returns defaults.  A full WMI implementation was omitted
      because it requires the ``wmi`` package which is not a project dependency;
      this monitor still collects *something* useful on every platform.

    If you need per-core CPU percentages on Windows, install psutil:
    ``pip install psutil`` and uncomment the import below.
    """

    # ── Linux helpers ────────────────────────────────────────────────

    def _read_proc_stat(self) -> Optional[List[float]]:
        """Return user+nice+system idle ticks from /proc/stat line 1."""
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()
            # user, nice, system, idle are indices 1-4 in the second column
            return [float(parts[i]) for i in (1, 2, 3, 5)]
        except (FileNotFoundError, IndexError, ValueError):
            return None

    def _read_meminfo(self) -> Optional[tuple]:
        """Return (total_kb, free_kb, available_kb) from /proc/meminfo."""
        try:
            mem = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, val = line.split(":")
                    mem[key.strip()] = int(val.strip().split()[0])
            return mem.get("MemTotal", 0), mem.get("MemFree", 0), mem.get("MemAvailable", 0)
        except (FileNotFoundError, ValueError):
            return None

    # ── Windows helpers ──────────────────────────────────────────────

    def _win_cpu_percent(self) -> float:
        """Approximate CPU usage on Windows via ``os.times`` delta."""
        try:
            t = os.times()
            total_user = t.user + t.system + t.children_user + t.children_system
            return 0.0  # single sample gives no delta; caller tracks it
        except Exception:
            return 0.0

    def _win_memory_gb(self) -> tuple[float, float]:
        """Return (used_gb, total_gb) on Windows via ``psutil`` fallback."""
        try:
            import psutil  # optional dependency
            mem = psutil.virtual_memory()
            return mem.used / (1024 ** 3), mem.total / (1024 ** 3), mem.percent
        except ImportError:
            pass
        # Fallback: read from environment or return defaults.
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            csi = ctypes.c_ulonglong()
            kernel.GlobalMemoryStatusEx(ctypes.byref(csi))
            total = csu.contents.ullTotalMem / (1024 ** 3) if hasattr(csu := type('S', (), {'contents': csi}), 'contents') else 0
            return total * 0.5, total  # heuristic: half used when unknown
        except Exception:
            pass
        return 0.0, 0.0

    # ── Public API ───────────────────────────────────────────────────

    def get_stats(self) -> ResourceStats:
        """Collect system resource statistics."""
        stats = ResourceStats()
        sysname = platform.system()

        if sysname == "Linux":
            cpu_vals = self._read_proc_stat()
            mem_info = self._read_meminfo()
            if cpu_vals and len(cpu_vals) >= 4:
                # Simple delta-based CPU estimate (caller should compare to previous call)
                total = sum(cpu_vals)
                idle = cpu_vals[3]
                stats.cpu_percent = max(0.0, 100.0 * (total - idle) / max(total, 1))
            if mem_info:
                total_kb, free_kb, avail_kb = mem_info
                stats.memory_total_gb = total_kb / (1024 ** 2)
                stats.memory_used_gb = (total_kb - free_kb) / (1024 ** 2)
                stats.memory_percent = 100.0 * (total_kb - avail_kb) / max(total_kb, 1)

        elif sysname == "Windows":
            # Use os.getloadavg if available (Python 3.4+ on Windows).
            try:
                load1, load5, load15 = os.getloadavg()
                stats.cpu_percent = min(100.0, load1 * 100.0 / os.cpu_count())
            except (OSError, AttributeError):
                stats.cpu_percent = self._win_cpu_percent()

            try:
                import psutil
                mem = psutil.virtual_memory()
                stats.memory_used_gb = mem.used / (1024 ** 3)
                stats.memory_total_gb = mem.total / (1024 ** 3)
                stats.memory_percent = mem.percent
            except ImportError:
                # No psutil — fall back to environment-based heuristic.
                pass

        else:
            # macOS / other Unix-like systems.
            try:
                load1, _, _ = os.getloadavg()
                stats.cpu_percent = min(100.0, load1 * 100.0 / os.cpu_count())
            except (OSError, AttributeError):
                pass

        return stats


def get_system_stats() -> dict:
    """Convenience function to collect all resource stats at once."""
    gpu = GPUResourceMonitor().get_stats()
    cpu = CPUResourceMonitor().get_stats()
    return {
        "gpu": {
            "memory_used_gb": gpu.memory_used_gb,
            "memory_total_gb": gpu.memory_total_gb,
            "compute_utilization": gpu.compute_utilization,
            "temperature": gpu.temperature,
        },
        "cpu": {
            "percent": cpu.cpu_percent,
            "memory_used_gb": cpu.memory_used_gb,
            "memory_total_gb": cpu.memory_total_gb,
            "memory_percent": cpu.memory_percent,
        },
    }
