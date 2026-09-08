"""Ensemble methods for CR-Pipeline.

Provides:
- Model ensembling strategies
- Weight averaging
- Stacking
- Diversity measurement
- Ensemble optimization
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Ensemble Methods
# =============================================================================


class EnsembleMethod:
    """Base class for ensemble methods."""

    def combine(self, models: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine multiple models into one.

        Args:
            models: List of model weight arrays.
            weights: Optional combination weights.

        Returns:
            Combined model weights.
        """
        raise NotImplementedError


class WeightAveragingEnsemble(EnsembleMethod):
    """Simple weight averaging ensemble.

    Averages weights across models, optionally weighted by performance.
    """

    def __init__(self, use_performance_weighting: bool = True):
        """Initialize.

        Args:
            use_performance_weighting: Use fitness-based weights.
        """
        self.use_performance_weighting = use_performance_weighting

    def combine(self, models: List[np.ndarray],
                fitnesses: Optional[List[float]] = None,
                weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine models by weighted averaging.

        Args:
            models: List of model weight arrays (all same shape).
            fitnesses: Fitness scores for each model.
            weights: Explicit combination weights.

        Returns:
            Combined weight array.
        """
        if not models:
            return np.array([])

        # Compute or use provided weights
        if weights is None and fitnesses is not None and self.use_performance_weighting:
            # Softmax weighting based on fitness
            fitnesses = np.array(fitnesses, dtype=float)
            fitnesses -= fitnesses.max()  # Numerical stability
            weights = np.exp(fitnesses)
            weights /= weights.sum()
        elif weights is None:
            weights = np.ones(len(models)) / len(models)
        else:
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()

        # Weighted average
        combined = np.zeros_like(models[0])
        for w, model in zip(weights, models):
            combined += w * model

        return combined


class GeometricMeanEnsemble(EnsembleMethod):
    """Geometric mean ensemble.

    Uses geometric mean of weights, better for preserving sparsity.
    """

    def combine(self, models: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine models using geometric mean."""
        if not models:
            return np.array([])

        # Convert to log space
        log_models = [np.log(np.abs(m) + 1e-10) for m in models]

        # Average in log space
        avg_log = np.mean(log_models, axis=0)

        # Convert back
        combined = np.sign(models[0]) * np.exp(avg_log)

        return combined


class StackingEnsemble(EnsembleMethod):
    """Stacking ensemble with meta-learner.

    Trains a meta-learner to combine model predictions.
    """

    def __init__(self, meta_learner_type: str = "linear"):
        """Initialize.

        Args:
            meta_learner_type: Type of meta-learner ("linear", "ridge", "neural").
        """
        self.meta_learner_type = meta_learner_type
        self.meta_weights: Optional[np.ndarray] = None

    def fit(self, model_outputs: List[np.ndarray],
            targets: np.ndarray) -> np.ndarray:
        """Train the meta-learner.

        Args:
            model_outputs: List of model prediction arrays.
            targets: Ground truth targets.

        Returns:
            Meta-learner weights.
        """
        # Stack outputs
        if self.meta_learner_type == "linear":
            # Simple linear combination
            X = np.column_stack([o.flatten() for o in model_outputs])
            y = targets.flatten()

            # Normal equations: w = (X^T X)^-1 X^T y
            XtX = X.T @ X
            Xty = X.T @ y

            # Add regularization
            XtX += 1e-6 * np.eye(XtX.shape[0])

            self.meta_weights = np.linalg.solve(XtX, Xty)
            return self.meta_weights

        elif self.meta_learner_type == "ridge":
            # Ridge regression
            X = np.column_stack([o.flatten() for o in model_outputs])
            y = targets.flatten()
            alpha = 1.0

            XtX = X.T @ X + alpha * np.eye(X.shape[1])
            Xty = X.T @ y

            self.meta_weights = np.linalg.solve(XtX, Xty)
            return self.meta_weights

        else:
            logger.warning(f"Unknown meta-learner type: {self.meta_learner_type}")
            return np.ones(len(model_outputs)) / len(model_outputs)

    def combine(self, models: List[np.ndarray],
                weights: Optional[List[float]] = None) -> np.ndarray:
        """Combine models using meta-learner weights."""
        if self.meta_weights is None:
            logger.warning("Meta-learner not fitted, using equal weights")
            return WeightAveragingEnsemble().combine(models)

        # Apply meta-learner weights
        combined = sum(w * m for w, m in zip(self.meta_weights, models))
        return combined / len(models)


# =============================================================================
# Diversity Metrics
# =============================================================================


class DiversityMetric:
    """Compute diversity between models."""

    @staticmethod
    def pairwise_distance(models: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise distances between models.

        Args:
            models: List of model weight arrays.

        Returns:
            Pairwise distance matrix.
        """
        n = len(models)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(models[i] - models[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    @staticmethod
    def disagreement_rate(predictions: List[np.ndarray]) -> float:
        """Compute average disagreement rate between models.

        Args:
            predictions: List of model prediction arrays.

        Returns:
            Average pairwise disagreement rate.
        """
        n = len(predictions)
        if n < 2:
            return 0.0

        total_disagreement = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Binary disagreement
                pred_i = np.argmax(predictions[i], axis=-1) if predictions[i].ndim > 1 else predictions[i]
                pred_j = np.argmax(predictions[j], axis=-1) if predictions[j].ndim > 1 else predictions[j]

                disagreement = np.mean(pred_i != pred_j)
                total_disagreement += disagreement
                count += 1

        return float(total_disagreement / count) if count > 0 else 0.0

    @staticmethod
    def correlation_matrix(predictions: List[np.ndarray]) -> np.ndarray:
        """Compute correlation matrix between model predictions.

        Args:
            predictions: List of model prediction arrays.

        Returns:
            Correlation matrix.
        """
        n = len(predictions)
        corr_matrix = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(predictions[i].flatten(), predictions[j].flatten())[0, 1]
                if not np.isnan(corr):
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        return corr_matrix


# =============================================================================
# Ensemble Optimizer
# =============================================================================


class EnsembleOptimizer:
    """Optimize ensemble combinations.

    Finds optimal model selection and weighting for best ensemble performance.
    """

    def __init__(self, n_iterations: int = 50, seed: int = 42):
        """Initialize.

        Args:
            n_iterations: Number of optimization iterations.
            seed: Random seed.
        """
        self.n_iterations = n_iterations
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def optimize(self, models: List[np.ndarray],
                 validation_fn,
                 n_models_to_select: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Optimize ensemble combination.

        Args:
            models: List of model weight arrays.
            validation_fn: Function(model_weights, validation_data) -> score.
            n_models_to_select: Number of models to select. None for all.

        Returns:
            Tuple of (best_weights, optimization_info).
        """
        best_score = -float("inf")
        best_weights = np.ones(len(models)) / len(models)
        info = {"score_history": [], "diversity_history": []}

        for i in range(self.n_iterations):
            # Sample combination weights
            if n_models_to_select:
                # Select subset
                indices = self.rng.choice(len(models), size=n_models_to_select, replace=False)
                weights = np.zeros(len(models))
                weights[indices] = 1.0 / n_models_to_select
            else:
                # Random weights
                weights = self.rng.dirichlet(np.ones(len(models)))

            # Evaluate
            score = validation_fn(weights)

            info["score_history"].append({
                "iteration": i,
                "score": float(score),
                "n_models": int(np.sum(weights > 0.01)),
            })

            if score > best_score:
                best_score = score
                best_weights = weights.copy()

            # Track diversity
            active_models = [models[j] for j in range(len(models)) if weights[j] > 0.01]
            if len(active_models) >= 2:
                dist = DiversityMetric.pairwise_distance(active_models)
                info["diversity_history"].append(float(np.mean(dist)))

        info["best_score"] = float(best_score)
        info["n_models_used"] = int(np.sum(best_weights > 0.01))

        return best_weights, info


# =============================================================================
# Ensemble Builder
# =============================================================================


@dataclass
class EnsembleResult:
    """Result of ensemble building."""
    combined_weights: np.ndarray
    member_weights: List[np.ndarray]
    member_fitnesses: List[float]
    ensemble_score: float
    diversity_score: float
    method: str
    optimization_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ensemble_score": self.ensemble_score,
            "diversity_score": self.diversity_score,
            "method": self.method,
            "n_members": len(self.member_weights),
            "member_fitnesses": self.member_fitnesses,
            "optimization_info": self.optimization_info,
        }


class EnsembleBuilder:
    """Build and optimize ensembles."""

    def __init__(self):
        """Initialize the ensemble builder."""
        self._history: List[EnsembleResult] = []

    def build_ensemble(
        self,
        models: List[np.ndarray],
        fitnesses: List[float],
        method: str = "weighted_average",
        optimize: bool = True,
        n_iterations: int = 50,
    ) -> EnsembleResult:
        """Build an ensemble from multiple models.

        Args:
            models: List of model weight arrays.
            fitnesses: Fitness scores for each model.
            method: Ensemble method ("weighted_average", "geometric_mean", "stacking").
            optimize: Whether to optimize combination weights.
            n_iterations: Optimization iterations.

        Returns:
            EnsembleResult.
        """
        # Create ensemble method
        if method == "weighted_average":
            ensemble = WeightAveragingEnsemble(use_performance_weighting=optimize)
        elif method == "geometric_mean":
            ensemble = GeometricMeanEnsemble()
        elif method == "stacking":
            ensemble = StackingEnsemble()
        else:
            ensemble = WeightAveragingEnsemble()

        # Get combination weights
        if optimize:
            optimizer = EnsembleOptimizer(n_iterations=n_iterations)

            def validation_fn(weights):
                combined = ensemble.combine(models, weights=weights)
                # Simple fitness proxy
                return np.mean(fitnesses) * (1 + np.linalg.norm(weights) / np.sqrt(len(weights)))

            weights, opt_info = optimizer.optimize(models, validation_fn)
        else:
            weights = None
            opt_info = {}

        # Combine
        combined = ensemble.combine(models, fitnesses=fitnesses, weights=weights)

        # Compute diversity
        diversity = DiversityMetric.disagreement_rate(
            [np.random.randn(100) for _ in models]  # Placeholder predictions
        )

        result = EnsembleResult(
            combined_weights=combined,
            member_weights=list(models),
            member_fitnesses=list(fitnesses),
            ensemble_score=float(np.mean(fitnesses)),
            diversity_score=float(diversity),
            method=method,
            optimization_info=opt_info,
        )

        self._history.append(result)
        return result

    def get_history(self) -> List[dict]:
        """Get ensemble building history."""
        return [r.to_dict() for r in self._history]
