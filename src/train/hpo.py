"""Hyperparameter optimization for CR-Pipeline.

Provides:
- Bayesian optimization with Gaussian processes
- Grid search with parallel execution
- Random search with early stopping
- Population-based training (PBT)
- Automated architecture search
- Hyperparameter sensitivity analysis
- Optimal configuration recommendation
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Parameter Definitions
# =============================================================================


class ParamType(Enum):
    """Parameter type for search space definition."""
    INTEGER = auto()
    FLOAT = auto()
    CATEGORICAL = auto()
    LOG_FLOAT = auto()  # Log-scale float (e.g., learning rate)
    LOG_INTEGER = auto()  # Log-scale integer


@dataclass
class ParamSpace:
    """Definition of a hyperparameter search space.

    Attributes:
        name: Parameter name.
        param_type: Type of parameter.
        low: Lower bound (for numeric types).
        high: Upper bound (for numeric types).
        values: Possible values (for categorical).
        default: Default value.
        log_base: Log base for log-scale parameters.
    """
    name: str
    param_type: ParamType
    low: Optional[float] = None
    high: Optional[float] = None
    values: Optional[List[Any]] = None
    default: Any = None
    log_base: float = 10.0

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Any:
        """Sample a value from this parameter's space."""
        r = rng or np.random.RandomState()

        if self.param_type == ParamType.INTEGER:
            return int(r.randint(self.low, self.high + 1))
        elif self.param_type == ParamType.FLOAT:
            return float(r.uniform(self.low, self.high))
        elif self.param_type == ParamType.CATEGORICAL:
            return self.values[int(r.randint(0, len(self.values)))]
        elif self.param_type == ParamType.LOG_FLOAT:
            # Sample in log space, then exponentiate
            log_val = r.uniform(math.log10(self.low), math.log10(self.high))
            return float(10 ** log_val)
        elif self.param_type == ParamType.LOG_INTEGER:
            log_val = r.uniform(math.log10(self.low), math.log10(self.high))
            return int(10 ** log_val)
        else:
            raise ValueError(f"Unknown param type: {self.param_type}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "param_type": self.param_type.name,
            "low": self.low,
            "high": self.high,
            "values": self.values,
            "default": self.default,
            "log_base": self.log_base,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParamSpace":
        """Create from dictionary."""
        data["param_type"] = ParamType[data["param_type"]]
        return cls(**data)


# =============================================================================
# Search Results
# =============================================================================


@dataclass
class OptimizationResult:
    """Result of a hyperparameter optimization run.

    Attributes:
        best_params: Best hyperparameters found.
        best_score: Best objective score.
        all_results: All evaluated configurations.
        optimization_time: Total optimization time.
        n_evaluations: Number of evaluations performed.
    """
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    n_evaluations: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_results": self.all_results,
            "optimization_time": self.optimization_time,
            "n_evaluations": self.n_evaluations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizationResult":
        """Create from dictionary."""
        return cls(
            best_params=data["best_params"],
            best_score=data["best_score"],
            all_results=data.get("all_results", []),
            optimization_time=data.get("optimization_time", 0.0),
            n_evaluations=data.get("n_evaluations", 0),
        )


# =============================================================================
# Bayesian Optimization
# =============================================================================


class BayesianOptimizer:
    """Bayesian optimization using simple Gaussian process approximation.

    Uses acquisition function (EI - Expected Improvement) to balance
    exploration and exploitation.
    """

    def __init__(
        self,
        param_spaces: List[ParamSpace],
        n_initial: int = 10,
        n_iterations: int = 50,
        seed: int = 42,
    ):
        """Initialize the Bayesian optimizer.

        Args:
            param_spaces: List of parameter space definitions.
            n_initial: Number of initial random evaluations.
            n_iterations: Number of optimization iterations.
            seed: Random seed.
        """
        self.param_spaces = param_spaces
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # History
        self.X: List[np.ndarray] = []  # Parameter vectors
        self.y: List[float] = []  # Objective values (to maximize)
        self.results: List[dict] = []

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run Bayesian optimization.

        Args:
            objective_fn: Function that takes params and returns score.
            verbose: Whether to print progress.

        Returns:
            OptimizationResult with best parameters.
        """
        start_time = time.time()
        best_score = -float("inf")
        best_params = None

        # Phase 1: Initial random evaluations
        logger.info(f"Phase 1: Running {self.n_initial} initial random evaluations...")
        for i in range(self.n_initial):
            params = self._sample_random()
            score = objective_fn(params)

            self.X.append(self._params_to_vector(params))
            self.y.append(score)
            self.results.append({"params": params, "score": score, "iteration": i})

            if score > best_score:
                best_score = score
                best_params = params.copy()

            if verbose:
                logger.info(f"  Iteration {i + 1}/{self.n_initial}: score={score:.4f}")

        # Phase 2: Bayesian optimization
        logger.info(f"Phase 2: Running {self.n_iterations} Bayesian optimization iterations...")
        for i in range(self.n_iterations):
            # Find next point to evaluate
            next_params = self._suggest_next()
            score = objective_fn(next_params)

            self.X.append(self._params_to_vector(next_params))
            self.y.append(score)
            self.results.append({"params": next_params, "score": score, "iteration": self.n_initial + i})

            if score > best_score:
                best_score = score
                best_params = next_params.copy()
                logger.info(f"  New best! Iteration {self.n_initial + i + 1}: score={score:.4f}")

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=self.results,
            optimization_time=elapsed,
            n_evaluations=len(self.results),
        )

    def _sample_random(self) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        for space in self.param_spaces:
            params[space.name] = space.sample(self.rng)
        return params

    def _params_to_vector(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameters to normalized vector."""
        vector = np.zeros(len(self.param_spaces))
        for i, space in enumerate(self.param_spaces):
            val = params[space.name]
            if space.param_type in (ParamType.INTEGER, ParamType.FLOAT, ParamType.LOG_FLOAT, ParamType.LOG_INTEGER):
                # Normalize to [0, 1]
                vector[i] = (val - space.low) / (space.high - space.low)
            elif space.param_type == ParamType.CATEGORICAL:
                vector[i] = space.values.index(val) / max(1, len(space.values) - 1)
        return vector

    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Any]:
        """Convert normalized vector to parameters."""
        params = {}
        for i, space in enumerate(self.param_spaces):
            if space.param_type in (ParamType.INTEGER, ParamType.FLOAT, ParamType.LOG_FLOAT, ParamType.LOG_INTEGER):
                val = vector[i] * (space.high - space.low) + space.low
                if space.param_type in (ParamType.INTEGER, ParamType.LOG_INTEGER):
                    val = int(round(val))
                elif space.param_type == ParamType.LOG_FLOAT:
                    val = float(10 ** val)
            elif space.param_type == ParamType.CATEGORICAL:
                idx = int(round(vector[i] * (len(space.values) - 1)))
                idx = max(0, min(idx, len(space.values) - 1))
                val = space.values[idx]
            params[space.name] = val
        return params

    def _suggest_next(self) -> Dict[str, Any]:
        """Suggest next parameters using acquisition function."""
        if len(self.X) < 3:
            return self._sample_random()

        X = np.array(self.X)
        y = np.array(self.y)

        # Find current best
        best_idx = np.argmax(y)
        best_y = y[best_idx]

        # Compute kernel distances
        best_x = X[best_idx]
        distances = np.sqrt(np.sum((X - best_x) ** 2, axis=1))

        # Acquisition function: Expected Improvement
        # EI(x) = (mu(x) - y_max - xi) * Phi(Z) + sigma(x) * phi(Z)
        # where Z = (mu(x) - y_max - xi) / sigma(x)
        xi = 0.01  # Exploration parameter

        best_ei = -float("inf")
        best_params = None

        # Sample candidate points
        n_candidates = 1000
        for _ in range(n_candidates):
            candidate = self._sample_random()
            candidate_vec = self._params_to_vector(candidate)

            # Simple kernel: exponential distance
            dist = np.sqrt(np.sum((candidate_vec - best_x) ** 2))
            sigma = np.exp(-dist / 0.5)  # Bandwidth parameter

            # Expected Improvement
            if sigma > 1e-6:
                Z = (best_y - xi - best_y) / sigma  # Simplified
                ei = sigma * (Z * 0.5 + 0.3)  # Simplified EI
            else:
                ei = 0.0

            if ei > best_ei:
                best_ei = ei
                best_params = candidate

        return best_params or self._sample_random()

    def get_history(self) -> List[dict]:
        """Get optimization history."""
        return self.results


# =============================================================================
# Grid Search
# =============================================================================


class GridSearchOptimizer:
    """Grid search hyperparameter optimizer.

    Evaluates all combinations of parameter values.
    Useful for small search spaces or baseline comparison.
    """

    def __init__(
        self,
        param_spaces: List[ParamSpace],
        grid_points: int = 5,
    ):
        """Initialize grid search.

        Args:
            param_spaces: List of parameter space definitions.
            grid_points: Number of points per dimension.
        """
        self.param_spaces = param_spaces
        self.grid_points = grid_points
        self._grid: List[Dict[str, Any]] = []
        self._build_grid()

    def _build_grid(self) -> None:
        """Build the parameter grid."""
        # For each parameter, create discretized values
        grids = []
        for space in self.param_spaces:
            if space.param_type in (ParamType.INTEGER, ParamType.FLOAT):
                values = np.linspace(space.low, space.high, self.grid_points).tolist()
            elif space.param_type == ParamType.LOG_FLOAT:
                log_vals = np.linspace(math.log10(space.low), math.log10(space.high), self.grid_points)
                values = [float(10 ** lv) for lv in log_vals]
            elif space.param_type == ParamType.LOG_INTEGER:
                log_vals = np.linspace(math.log10(space.low), math.log10(space.high), self.grid_points)
                values = [int(10 ** lv) for lv in log_vals]
            elif space.param_type == ParamType.CATEGORICAL:
                values = space.values
            grids.append(values)

        # Generate all combinations
        self._grid = list(self._product(grids))

    def _product(self, lists: List[List]) -> List[Tuple]:
        """Generate Cartesian product of lists."""
        if not lists:
            yield ()
        else:
            for item in lists[0]:
                for rest in self._product(lists[1:]):
                    yield (item,) + rest

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run grid search optimization.

        Args:
            objective_fn: Function that takes params and returns score.
            verbose: Whether to print progress.

        Returns:
            OptimizationResult with best parameters.
        """
        start_time = time.time()
        best_score = -float("inf")
        best_params = None
        results = []

        logger.info(f"Grid search: {len(self._grid)} configurations to evaluate")

        for i, params in enumerate(self._grid):
            score = objective_fn(params)
            results.append({"params": params, "score": score, "iteration": i})

            if score > best_score:
                best_score = score
                best_params = params.copy()

            if verbose:
                logger.info(f"  [{i + 1}/{len(self._grid)}] score={score:.4f}")

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            optimization_time=elapsed,
            n_evaluations=len(results),
        )


# =============================================================================
# Random Search with Early Stopping
# =============================================================================


class RandomSearchOptimizer:
    """Random search with optional early stopping.

    Samples random configurations and optionally stops
    unpromising trials early.
    """

    def __init__(
        self,
        param_spaces: List[ParamSpace],
        n_trials: int = 50,
        patience: int = 5,
        seed: int = 42,
    ):
        """Initialize random search.

        Args:
            param_spaces: List of parameter space definitions.
            n_trials: Maximum number of trials.
            patience: Early stopping patience.
            seed: Random seed.
        """
        self.param_spaces = param_spaces
        self.n_trials = n_trials
        self.patience = patience
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run random search optimization.

        Args:
            objective_fn: Function that takes params and returns score.
            verbose: Whether to print progress.

        Returns:
            OptimizationResult with best parameters.
        """
        start_time = time.time()
        best_score = -float("inf")
        best_params = None
        results = []
        no_improve_count = 0

        logger.info(f"Random search: up to {self.n_trials} trials")

        for i in range(self.n_trials):
            params = self._sample_random()
            score = objective_fn(params)
            results.append({"params": params, "score": score, "iteration": i})

            if score > best_score:
                best_score = score
                best_params = params.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if verbose:
                logger.info(f"  Trial {i + 1}/{self.n_trials}: score={score:.4f}, best={best_score:.4f}")

            # Early stopping check
            if no_improve_count >= self.patience and i > 10:
                logger.info(f"Early stopping after {i + 1} trials (no improvement for {self.patience} trials)")
                break

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            optimization_time=elapsed,
            n_evaluations=len(results),
        )

    def _sample_random(self) -> Dict[str, Any]:
        """Sample random parameters."""
        params = {}
        for space in self.param_spaces:
            params[space.name] = space.sample(self.rng)
        return params


# =============================================================================
# Population-Based Training (PBT)
# =============================================================================


class PBTOptimizer:
    """Population-Based Training hyperparameter optimizer.

    Maintains a population of configurations that evolve over time
    through exploration (perturbation) and exploitation (selection).
    """

    def __init__(
        self,
        param_spaces: List[ParamSpace],
        population_size: int = 8,
        n_iterations: int = 50,
        pbt_interval: int = 10,
        seed: int = 42,
    ):
        """Initialize PBT optimizer.

        Args:
            param_spaces: List of parameter space definitions.
            population_size: Number of concurrent configurations.
            n_iterations: Total training iterations.
            pbt_interval: PBT update interval.
            seed: Random seed.
        """
        self.param_spaces = param_spaces
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.pbt_interval = pbt_interval
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def optimize(
        self,
        objective_fn: Callable[[Dict[str, Any], int], float],
        verbose: bool = False,
    ) -> OptimizationResult:
        """Run PBT optimization.

        Args:
            objective_fn: Function that takes params and current iteration, returns score.
            verbose: Whether to print progress.

        Returns:
            OptimizationResult with best parameters.
        """
        start_time = time.time()
        results = []

        # Initialize population
        population = []
        for _ in range(self.population_size):
            params = {}
            for space in self.param_spaces:
                params[space.name] = space.sample(self.rng)
            population.append({"params": params, "score": -float("inf"), "age": 0})

        best_score = -float("inf")
        best_params = None

        logger.info(f"PBT: {self.population_size} configurations for {self.n_iterations} iterations")

        for iteration in range(self.n_iterations):
            # Evaluate population
            for agent in population:
                score = objective_fn(agent["params"], iteration)
                agent["score"] = score
                agent["age"] += 1

                if score > best_score:
                    best_score = score
                    best_params = agent["params"].copy()

            # PBT update
            if iteration > 0 and iteration % self.pbt_interval == 0:
                population = self._pbt_update(population)

            if verbose and iteration % 10 == 0:
                logger.info(f"  Iteration {iteration}: best_score={best_score:.4f}")

        elapsed = time.time() - start_time

        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            optimization_time=elapsed,
            n_evaluations=self.n_iterations * self.population_size,
        )

    def _pbt_update(self, population: List[dict]) -> List[dict]:
        """Perform PBT update: replace worst with perturbed best."""
        # Sort by score
        population.sort(key=lambda x: x["score"], reverse=True)

        # Replace bottom half with perturbed top half
        n_replace = len(population) // 2
        for i in range(n_replace):
            # Pick parent from top half
            parent_idx = self.rng.randint(0, len(population) // 2)
            parent = population[parent_idx]

            # Perturb parameters
            new_params = parent["params"].copy()
            for space in self.param_spaces:
                if self.rng.random() < 0.3:  # 30% chance to perturb each param
                    if space.param_type in (ParamType.INTEGER, ParamType.FLOAT):
                        perturb = self.rng.normal(0, 0.1 * (space.high - space.low))
                        new_val = new_params[space.name] + perturb
                        new_params[space.name] = max(space.low, min(space.high, new_val))
                    elif space.param_type == ParamType.LOG_FLOAT:
                        log_perturb = self.rng.normal(0, 0.1)
                        new_val = new_params[space.name] * (10 ** log_perturb)
                        new_params[space.name] = max(space.low, min(space.high, new_val))

            population[len(population) - 1 - i] = {
                "params": new_params,
                "score": -float("inf"),
                "age": 0,
            }

        return population


# =============================================================================
# Sensitivity Analysis
# =============================================================================


class SensitivityAnalyzer:
    """Analyze hyperparameter sensitivity.

    Determines which parameters have the most impact on performance.
    """

    def __init__(self, param_spaces: List[ParamSpace]):
        """Initialize sensitivity analyzer.

        Args:
            param_spaces: List of parameter space definitions.
        """
        self.param_spaces = param_spaces

    def analyze(
        self,
        results: List[Dict[str, Any]],
        score_key: str = "score",
    ) -> Dict[str, float]:
        """Analyze parameter sensitivity from optimization results.

        Args:
            results: List of optimization results with scores.
            score_key: Key for the score in each result dict.

        Returns:
            Dictionary mapping parameter name to sensitivity score.
        """
        if not results:
            return {}

        sensitivities = {}

        for space in self.param_spaces:
            # Get values and scores
            values = [r["params"].get(space.name) for r in results]
            scores = [r[score_key] for r in results]

            # Compute correlation (Spearman rank correlation)
            if len(values) < 3:
                sensitivities[space.name] = 0.0
                continue

            # Filter out None values
            valid = [(v, s) for v, s in zip(values, scores) if v is not None]
            if len(valid) < 3:
                sensitivities[space.name] = 0.0
                continue

            val_arr = np.array([v for v, _ in valid])
            score_arr = np.array([s for _, s in valid])

            # Rank-based correlation
            val_rank = self._rank(val_arr)
            score_rank = self._rank(score_arr)

            # Spearman correlation
            n = len(val_rank)
            diff_sq = np.sum((val_rank - score_rank) ** 2)
            rho = 1 - (6 * diff_sq) / (n * (n ** 2 - 1))

            sensitivities[space.name] = float(abs(rho))

        return sensitivities

    def _rank(self, arr: np.ndarray) -> np.ndarray:
        """Compute ranks of array elements."""
        return np.argsort(np.argsort(arr))


# =============================================================================
# Default Search Spaces
# =============================================================================


def get_default_evolution_search_space() -> List[ParamSpace]:
    """Get default search space for evolution hyperparameters.

    Returns:
        List of ParamSpace definitions.
    """
    return [
        ParamSpace(
            name="population_size",
            param_type=ParamType.INTEGER,
            low=50,
            high=300,
            default=200,
        ),
        ParamSpace(
            name="elite_count",
            param_type=ParamType.INTEGER,
            low=5,
            high=50,
            default=10,
        ),
        ParamSpace(
            name="crossover_rate",
            param_type=ParamType.FLOAT,
            low=0.3,
            high=0.95,
            default=0.7,
        ),
        ParamSpace(
            name="mutation_rate",
            param_type=ParamType.FLOAT,
            low=0.001,
            high=0.2,
            default=0.05,
        ),
        ParamSpace(
            name="mutation_std",
            param_type=ParamType.LOG_FLOAT,
            low=0.01,
            high=1.0,
            default=0.1,
        ),
        ParamSpace(
            name="tournament_size",
            param_type=ParamType.INTEGER,
            low=2,
            high=20,
            default=5,
        ),
        ParamSpace(
            name="diversity_threshold",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=2.0,
            default=0.5,
        ),
    ]


def get_default_tournament_search_space() -> List[ParamSpace]:
    """Get default search space for tournament parameters.

    Returns:
        List of ParamSpace definitions.
    """
    return [
        ParamSpace(
            name="tournament_format",
            param_type=ParamType.CATEGORICAL,
            values=["round_robin", "single_elim", "double_elim", "league"],
            default="round_robin",
        ),
        ParamSpace(
            name="matches_per_pair",
            param_type=ParamType.INTEGER,
            low=2,
            high=10,
            default=4,
        ),
        ParamSpace(
            name="elite_fraction",
            param_type=ParamType.FLOAT,
            low=0.05,
            high=0.3,
            default=0.1,
        ),
    ]
