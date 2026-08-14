"""CR-Pipeline: Model Deployment.

Provides:
- Model export to various formats (PyTorch, ONNX, TorchScript, NumPy, JSON)
- Model compression (pruning, quantization)
- Inference benchmarking
- Model versioning and registry
"""

from .export import (
    ModelExporter,
    ModelCompressor,
    InferenceBenchmarker,
    ModelFormat,
    ModelMetadata,
)

__all__ = [
    "ModelExporter",
    "ModelCompressor",
    "InferenceBenchmarker",
    "ModelFormat",
    "ModelMetadata",
]
