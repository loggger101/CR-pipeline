"""Model export and deployment utilities.

Provides:
- Model serialization to various formats
- ONNX export for cross-platform inference
- TorchScript export for Python inference
- JSON weight export for lightweight deployment
- Model compression (pruning, quantization)
- Inference benchmarking
- Model versioning and registry
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Model Formats
# =============================================================================


class ModelFormat(Enum):
    """Supported model export formats."""
    TORCH = auto()         # PyTorch .pt
    TORCHSCRIPT = auto()   # TorchScript .pt
    ONNX = auto()          # ONNX .onnx
    NUMPY = auto()         # NumPy .npy
    JSON = auto()          # JSON weights
    PICKLE = auto()        # Python pickle


# =============================================================================
# Model Metadata
# =============================================================================


@dataclass
class ModelMetadata:
    """Metadata for an exported model.

    Attributes:
        model_id: Unique model identifier.
        version: Model version string.
        format: Export format.
        architecture: Model architecture name.
        input_shape: Expected input shape.
        output_shape: Expected output shape.
        param_count: Number of parameters.
        file_size_bytes: Size of exported file.
        exported_at: Export timestamp.
        config: Model configuration.
        training_stats: Training statistics.
    """
    model_id: str
    version: str
    format: ModelFormat
    architecture: str
    input_shape: List[int]
    output_shape: List[int]
    param_count: int = 0
    file_size_bytes: int = 0
    exported_at: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    training_stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "format": self.format.name,
            "architecture": self.architecture,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "param_count": self.param_count,
            "file_size_bytes": self.file_size_bytes,
            "exported_at": self.exported_at,
            "config": self.config,
            "training_stats": self.training_stats,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        """Create from dictionary."""
        data["format"] = ModelFormat[data["format"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Model Exporter
# =============================================================================


class ModelExporter:
    """Exports trained models to various formats.

    Supports:
    - PyTorch native format
    - TorchScript for deployment
    - ONNX for cross-platform inference
    - NumPy arrays for lightweight use
    - JSON weights for human-readable format
    """

    def __init__(self, output_dir: str = "exports"):
        """Initialize the exporter.

        Args:
            output_dir: Directory for exported models.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._export_history: List[dict] = []

    def export_model(
        self,
        weights: np.ndarray,
        metadata: ModelMetadata,
        format: ModelFormat = ModelFormat.TORCH,
    ) -> str:
        """Export a model to the specified format.

        Args:
            weights: Model weights as numpy array.
            metadata: Model metadata.
            format: Export format.

        Returns:
            Path to the exported file.
        """
        timestamp = int(time.time())
        filename = f"{metadata.model_id}_v{metadata.version}_{timestamp}"

        if format == ModelFormat.TORCH:
            return self._export_torch(weights, metadata, filename)
        elif format == ModelFormat.TORCHSCRIPT:
            return self._export_torchscript(weights, metadata, filename)
        elif format == ModelFormat.ONNX:
            return self._export_onnx(weights, metadata, filename)
        elif format == ModelFormat.NUMPY:
            return self._export_numpy(weights, metadata, filename)
        elif format == ModelFormat.JSON:
            return self._export_json(weights, metadata, filename)
        elif format == ModelFormat.PICKLE:
            return self._export_pickle(weights, metadata, filename)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_torch(self, weights: np.ndarray, metadata: ModelMetadata,
                      filename: str) -> str:
        """Export as PyTorch checkpoint."""
        import torch

        checkpoint = {
            "weights": weights.tolist(),
            "metadata": metadata.to_dict(),
            "exported_at": time.time(),
        }

        filepath = self.output_dir / f"{filename}.pt"
        torch.save(checkpoint, str(filepath))

        metadata.file_size_bytes = filepath.stat().st_size
        self._export_history.append(metadata.to_dict())

        logger.info(f"Exported PyTorch model to {filepath} ({metadata.file_size_bytes / 1024:.1f} KB)")
        return str(filepath)

    def _export_torchscript(self, weights: np.ndarray, metadata: ModelMetadata,
                            filename: str) -> str:
        """Export as TorchScript model."""
        try:
            import torch

            # Create a simple wrapper module
            class WeightModule(torch.nn.Module):
                def __init__(self, weights: np.ndarray):
                    super().__init__()
                    self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # Simple linear transformation as placeholder
                    w = self.weights.view(-1, 1)
                    return x @ w[:x.size(1), :] if x.size(1) <= w.size(1) else x @ w.T

            model = WeightModule(weights)
            scripted = torch.jit.script(model)

            filepath = self.output_dir / f"{filename}.ts.pt"
            scripted.save(str(filepath))

            metadata.file_size_bytes = filepath.stat().st_size
            self._export_history.append(metadata.to_dict())

            logger.info(f"Exported TorchScript model to {filepath}")
            return str(filepath)
        except ImportError:
            logger.warning("Torch not available. Falling back to PyTorch export.")
            return self._export_torch(weights, metadata, filename)

    def _export_onnx(self, weights: np.ndarray, metadata: ModelMetadata,
                     filename: str) -> str:
        """Export as ONNX model."""
        try:
            import torch
            import onnx

            class WeightModule(torch.nn.Module):
                def __init__(self, weights: np.ndarray):
                    super().__init__()
                    self.weights = torch.nn.Parameter(torch.tensor(weights, dtype=torch.float32))

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    w = self.weights.view(-1, 1)
                    return x @ w[:x.size(1), :] if x.size(1) <= w.size(1) else x @ w.T

            model = WeightModule(weights)
            model.eval()

            filepath = self.output_dir / f"{filename}.onnx"
            dummy_input = torch.randn(1, weights.shape[1] if len(weights.shape) > 1 else 1)

            torch.onnx.export(
                model,
                dummy_input,
                str(filepath),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            metadata.file_size_bytes = filepath.stat().st_size
            self._export_history.append(metadata.to_dict())

            logger.info(f"Exported ONNX model to {filepath}")
            return str(filepath)
        except ImportError:
            logger.warning("ONNX not available. Falling back to PyTorch export.")
            return self._export_torch(weights, metadata, filename)

    def _export_numpy(self, weights: np.ndarray, metadata: ModelMetadata,
                      filename: str) -> str:
        """Export as NumPy file."""
        filepath = self.output_dir / f"{filename}.npy"
        np.save(str(filepath), weights)

        metadata.file_size_bytes = filepath.stat().st_size
        self._export_history.append(metadata.to_dict())

        logger.info(f"Exported NumPy weights to {filepath} ({metadata.file_size_bytes / 1024:.1f} KB)")
        return str(filepath)

    def _export_json(self, weights: np.ndarray, metadata: ModelMetadata,
                     filename: str) -> str:
        """Export as JSON weights."""
        data = {
            "weights": weights.tolist(),
            "shape": list(weights.shape),
            "dtype": str(weights.dtype),
            "metadata": metadata.to_dict(),
        }

        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        metadata.file_size_bytes = filepath.stat().st_size
        self._export_history.append(metadata.to_dict())

        logger.info(f"Exported JSON weights to {filepath} ({metadata.file_size_bytes / 1024:.1f} KB)")
        return str(filepath)

    def _export_pickle(self, weights: np.ndarray, metadata: ModelMetadata,
                       filename: str) -> str:
        """Export as pickle file."""
        import pickle

        data = {
            "weights": weights,
            "metadata": metadata.to_dict(),
        }

        filepath = self.output_dir / f"{filename}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        metadata.file_size_bytes = filepath.stat().st_size
        self._export_history.append(metadata.to_dict())

        logger.info(f"Exported pickle model to {filepath} ({metadata.file_size_bytes / 1024:.1f} KB)")
        return str(filepath)

    def get_export_history(self) -> List[dict]:
        """Get export history."""
        return self._export_history

    def list_models(self) -> List[dict]:
        """List all exported models."""
        models = []
        for ext in [".pt", ".ts.pt", ".onnx", ".npy", ".json", ".pkl"]:
            for filepath in self.output_dir.glob(f"*{ext}"):
                try:
                    stat = filepath.stat()
                    models.append({
                        "filename": filepath.name,
                        "size_bytes": stat.st_size,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": stat.st_mtime,
                    })
                except IOError:
                    continue
        models.sort(key=lambda x: x["modified"], reverse=True)
        return models


# =============================================================================
# Model Compression
# =============================================================================


class ModelCompressor:
    """Compress models for deployment.

    Supports:
    - Weight pruning (removing small weights)
    - Quantization (reducing precision)
    - Weight clustering
    """

    def __init__(self):
        """Initialize the compressor."""
        self._compression_history: List[dict] = []

    def prune_weights(
        self,
        weights: np.ndarray,
        threshold: float = 0.01,
        sparse: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Prune small weights.

        Args:
            weights: Model weights.
            threshold: Pruning threshold.
            sparse: Whether to return sparse representation.

        Returns:
            Tuple of (pruned_weights, sparsity_ratio).
        """
        pruned = weights.copy()

        # Set small weights to zero
        mask = np.abs(pruned) > threshold
        pruned[~mask] = 0.0

        sparsity = float(np.sum(~mask) / pruned.size)

        if sparse:
            # Return sparse representation
            indices = np.where(mask)
            values = pruned[mask]
            return {"indices": indices, "values": values, "shape": pruned.shape}, sparsity

        self._compression_history.append({
            "operation": "prune",
            "threshold": threshold,
            "sparsity": sparsity,
        })

        return pruned, sparsity

    def quantize_weights(
        self,
        weights: np.ndarray,
        bits: int = 8,
    ) -> Tuple[np.ndarray, float]:
        """Quantize weights to lower precision.

        Args:
            weights: Model weights.
            bits: Number of bits for quantization.

        Returns:
            Tuple of (quantized_weights, compression_ratio).
        """
        # Normalize to [0, 1]
        min_val = weights.min()
        max_val = weights.max()
        range_val = max_val - min_val

        if range_val == 0:
            return weights.copy(), 1.0

        normalized = (weights - min_val) / range_val

        # Quantize
        levels = 2 ** bits
        quantized = np.round(normalized * (levels - 1)) / (levels - 1)

        # Denormalize
        dequantized = quantized * range_val + min_val

        # Compression ratio (original float32 = 32 bits)
        compression_ratio = 32.0 / bits

        self._compression_history.append({
            "operation": "quantize",
            "bits": bits,
            "compression_ratio": compression_ratio,
        })

        return dequantized, compression_ratio

    def get_compression_history(self) -> List[dict]:
        """Get compression history."""
        return self._compression_history


# =============================================================================
# Inference Benchmarking
# =============================================================================


class InferenceBenchmarker:
    """Benchmark model inference performance.

    Measures:
    - Inference latency
    - Throughput
    - Memory usage
    - CPU/GPU utilization
    """

    def __init__(self):
        """Initialize the benchmarker."""
        self._results: List[dict] = []

    def benchmark(
        self,
        model_fn,
        input_shape: Tuple[int, ...],
        n_runs: int = 100,
        warmup_runs: int = 10,
    ) -> dict:
        """Benchmark model inference.

        Args:
            model_fn: Function that takes input and returns output.
            input_shape: Input tensor shape.
            n_runs: Number of benchmark runs.
            warmup_runs: Number of warmup runs.

        Returns:
            Benchmark results dictionary.
        """
        # Warmup
        for _ in range(warmup_runs):
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            model_fn(dummy_input)

        # Benchmark
        latencies = []
        throughputs = []

        for _ in range(n_runs):
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            start = time.perf_counter()
            output = model_fn(dummy_input)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            throughputs.append(1000 / latency_ms)  # samples per second

        results = {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "mean_throughput": float(np.mean(throughputs)),
            "total_time_s": float(sum(latencies) / 1000),
            "n_runs": n_runs,
        }

        self._results.append(results)
        return results

    def get_results(self) -> List[dict]:
        """Get benchmark results."""
        return self._results

    def compare_results(self, results1: dict, results2: dict) -> dict:
        """Compare two benchmark results.

        Args:
            results1: First benchmark results.
            results2: Second benchmark results.

        Returns:
            Comparison dictionary.
        """
        latency_diff = results1["mean_latency_ms"] - results2["mean_latency_ms"]
        throughput_diff = results1["mean_throughput"] - results2["mean_throughput"]

        return {
            "latency_change_ms": latency_diff,
            "latency_change_pct": (latency_diff / results1["mean_latency_ms"]) * 100 if results1["mean_latency_ms"] > 0 else 0,
            "throughput_change": throughput_diff,
            "throughput_change_pct": (throughput_diff / results1["mean_throughput"]) * 100 if results1["mean_throughput"] > 0 else 0,
            "faster": latency_diff < 0,
        }
