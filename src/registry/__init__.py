"""Model versioning and registry for CR-Pipeline.

Provides:
- Model version tracking
- Model registry with metadata
- Model comparison utilities
- Model promotion (checkpoint -> candidate -> production)
- Model lineage tracking
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model deployment stages."""
    CHECKPOINT = auto()    # Training checkpoint
    CANDIDATE = auto()     # Evaluated and considered
    PRODUCTION = auto()    # Deployed for live use
    ARCHIVED = auto()      # Archived


@dataclass
class ModelVersion:
    """Represents a single model version.

    Attributes:
        model_id: Unique model identifier.
        version: Version string.
        stage: Current deployment stage.
        fitness: Best fitness achieved.
        tournament_rank: Tournament ranking (if applicable).
        architecture: Architecture name.
        param_count: Number of parameters.
        file_path: Path to model weights.
        metadata: Additional model metadata.
        created_at: Creation timestamp.
        tags: Model tags for categorization.
    """
    model_id: str
    version: str
    stage: ModelStage
    fitness: float = 0.0
    tournament_rank: int = 0
    architecture: str = ""
    param_count: int = 0
    file_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage.name,
            "fitness": self.fitness,
            "tournament_rank": self.tournament_rank,
            "architecture": self.architecture,
            "param_count": self.param_count,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        data["stage"] = ModelStage[data["stage"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ModelRegistry:
    """Registry for model versions with metadata and lifecycle management."""

    def __init__(self, registry_dir: str = "runs/model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing registry from disk."""
        registry_path = self.registry_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)
            for model_data in data.get("models", []):
                model = ModelVersion.from_dict(model_data)
                self.models[model.model_id] = model

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_path = self.registry_dir / "registry.json"
        with open(registry_path, "w") as f:
            json.dump({"models": [m.to_dict() for m in self.models.values()]}, f, indent=2)

    def register_model(
        self,
        model_id: str,
        version: str,
        fitness: float,
        architecture: str = "",
        param_count: int = 0,
        file_path: str = "",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Register a new model version.

        Args:
            model_id: Unique model identifier.
            version: Version string.
            fitness: Best fitness achieved.
            architecture: Architecture name.
            param_count: Number of parameters.
            file_path: Path to model weights.
            tags: Model tags.
            metadata: Additional metadata.

        Returns:
            Registered ModelVersion.
        """
        model = ModelVersion(
            model_id=model_id,
            version=version,
            stage=ModelStage.CHECKPOINT,
            fitness=fitness,
            architecture=architecture,
            param_count=param_count,
            file_path=file_path,
            metadata=metadata or {},
            tags=tags or [],
        )
        self.models[model_id] = model
        self._save_registry()
        logger.info(f"Registered model {model_id} v{version} (fitness={fitness:.2f})")
        return model

    def promote_model(self, model_id: str, stage: ModelStage) -> ModelVersion:
        """Promote a model to a new stage.

        Args:
            model_id: Model to promote.
            stage: Target stage.

        Returns:
            Updated ModelVersion.
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")
        model = self.models[model_id]
        model.stage = stage
        self._save_registry()
        logger.info(f"Promoted model {model_id} to {stage.name}")
        return model

    def get_best_models(
        self,
        stage: Optional[ModelStage] = None,
        limit: int = 10,
        tags: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        """Get top models by fitness.

        Args:
            stage: Filter by stage.
            limit: Maximum models to return.
            tags: Filter by tags.

        Returns:
            List of ModelVersion sorted by fitness (descending).
        """
        models = list(self.models.values())
        if stage:
            models = [m for m in models if m.stage == stage]
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]
        models.sort(key=lambda m: m.fitness, reverse=True)
        return models[:limit]

    def get_models_by_architecture(self, architecture: str) -> List[ModelVersion]:
        """Get all models with a specific architecture."""
        return [m for m in self.models.values() if m.architecture == architecture]

    def get_production_models(self) -> List[ModelVersion]:
        """Get all production models."""
        return [m for m in self.models.values() if m.stage == ModelStage.PRODUCTION]

    def compare_models(
        self,
        model_ids: List[str],
    ) -> Dict[str, Any]:
        """Compare multiple models by fitness and metadata.

        Args:
            model_ids: List of model IDs to compare.

        Returns:
            Comparison dictionary.
        """
        comparison = {}
        for model_id in model_ids:
            if model_id in self.models:
                model = self.models[model_id]
                comparison[model_id] = {
                    "version": model.version,
                    "fitness": model.fitness,
                    "architecture": model.architecture,
                    "param_count": model.param_count,
                    "stage": model.stage.name,
                    "tags": model.tags,
                }
        return comparison
