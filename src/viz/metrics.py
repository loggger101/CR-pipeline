"""Metrics computation for training visualization.

Provides:
- Fitness curve computation and smoothing
- Population statistics over time
- Pareto front computation
- Performance comparison tools
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class TrainingMetrics:
    """Computes and manages training metrics.

    Tracks:
    - Fitness curves (best, mean, median, min, max, std)
    - Population diversity over generations
    - Convergence detection
    - Performance comparisons
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "best": [],
            "mean": [],
            "median": [],
            "min": [],
            "max": [],
            "std": [],
            "diversity": [],
        }
        self.generation_labels: List[int] = []

    def update(self, fitnesses: List[float], diversity: float = 0.0,
               generation: Optional[int] = None) -> None:
        """Update metrics with new fitness data.

        Args:
            fitnesses: List of fitness scores for current generation.
            diversity: Population diversity metric.
            generation: Generation number (auto-incremented if None).
        """
        if generation is not None:
            self.generation_labels.append(generation)

        self.history["best"].append(max(fitnesses))
        self.history["mean"].append(float(np.mean(fitnesses)))
        self.history["median"].append(float(np.median(fitnesses)))
        self.history["min"].append(min(fitnesses))
        self.history["max"].append(max(fitnesses))
        self.history["std"].append(float(np.std(fitnesses)))
        self.history["diversity"].append(diversity)

    def get_fitness_curves(self) -> Dict[str, List[float]]:
        """Get fitness curves as lists.

        Returns:
            Dictionary mapping metric name to list of values.
        """
        return {k: list(v) for k, v in self.history.items()}

    def get_smoothed_fitness(self, window: int = 10,
                             metric: str = "best") -> List[float]:
        """Get smoothed fitness values using rolling window.

        Args:
            window: Window size for smoothing.
            metric: Which metric to smooth ("best", "mean", etc.).

        Returns:
            Smoothed fitness values.
        """
        values = self.history.get(metric, [])
        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(float(np.mean(values[start:i + 1])))

        return smoothed

    def get_pareto_front(self, fitnesses: List[float],
                         diversity_values: Optional[List[float]] = None) -> List[int]:
        """Compute the Pareto front of non-dominated solutions.

        If diversity_values is provided, optimizes for both fitness and diversity.
        Otherwise, only considers fitness.

        Args:
            fitnesses: List of fitness scores.
            diversity_values: Optional diversity values.

        Returns:
            List of indices that are on the Pareto front.
        """
        if diversity_values is None:
            # Single-objective: all points with max fitness
            max_fit = max(fitnesses)
            return [i for i, f in enumerate(fitnesses) if f >= max_fit * 0.99]

        # Multi-objective: fitness and diversity
        pareto = []
        for i in range(len(fitnesses)):
            dominated = False
            for j in range(len(fitnesses)):
                if i == j:
                    continue
                if (fitnesses[j] >= fitnesses[i] and
                        diversity_values[j] >= diversity_values[i] and
                        (fitnesses[j] > fitnesses[i] or
                         diversity_values[j] > diversity_values[i])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(i)

        return pareto

    def get_convergence_status(self, window: int = 20,
                               threshold: float = 1e-4) -> Dict:
        """Detect convergence based on fitness stability.

        Args:
            window: Window to check for stability.
            threshold: Minimum change to consider non-converged.

        Returns:
            Dictionary with convergence status.
        """
        best = self.history.get("best", [])
        if len(best) < window:
            return {"converged": False, "reason": "insufficient_data"}

        recent = best[-window:]
        change = max(recent) - min(recent)

        return {
            "converged": change < threshold,
            "recent_max": max(recent),
            "recent_min": min(recent),
            "recent_mean": float(np.mean(recent)),
            "change": change,
            "window": window,
        }

    def get_generation_range(self) -> Tuple[int, int]:
        """Get the generation range covered by the metrics.

        Returns:
            Tuple of (start_generation, end_generation).
        """
        if not self.generation_labels:
            return (0, 0)
        return (self.generation_labels[0], self.generation_labels[-1])

    def get_summary(self) -> Dict:
        """Get a summary of all metrics.

        Returns:
            Dictionary of summary statistics.
        """
        best = self.history.get("best", [])
        mean = self.history.get("mean", [])

        return {
            "total_generations": len(best),
            "best_fitness": max(best) if best else 0,
            "best_generation": best.index(max(best)) if best else 0,
            "current_best": best[-1] if best else 0,
            "final_mean": mean[-1] if mean else 0,
            "final_std": self.history.get("std", [0])[-1] if self.history.get("std") else 0,
            "final_diversity": self.history.get("diversity", [0])[-1] if self.history.get("diversity") else 0,
            "convergence": self.get_convergence_status(),
        }

    def add_from_history(self, history: Dict[str, List[float]]) -> None:
        """Add data from an existing fitness history.

        Args:
            history: Dictionary of metric name to list of values.
        """
        for key in ["best", "mean", "median", "min", "max", "std", "diversity"]:
            if key in history:
                self.history[key].extend(history[key])
