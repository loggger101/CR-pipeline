"""Metrics computation for training visualization.

Provides:
- Fitness curve computation and smoothing
- Population statistics over time
- Pareto front computation
- Performance comparison tools
- Advanced statistical analysis
- Convergence detection
- Growth rate analysis
- Statistical significance testing
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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

    # =============================================================================
    # Advanced Metrics
    # =============================================================================

    def get_growth_rate(self, window: int = 10) -> List[float]:
        """Compute the growth rate of best fitness over time.

        Args:
            window: Window size for computing rate of change.

        Returns:
            List of growth rates per generation.
        """
        best = self.history.get("best", [])
        if len(best) < window + 1:
            return [0.0] * len(best)

        rates = []
        for i in range(len(best)):
            if i < window:
                rates.append(0.0)
            else:
                start_val = np.mean(best[i-window:i])
                end_val = np.mean(best[max(0, i-window):i])
                rates.append(float((end_val - start_val) / window) if start_val != 0 else 0.0)

        return rates

    def get_acceleration(self, window: int = 10) -> List[float]:
        """Compute the acceleration (rate of change of growth rate).

        Args:
            window: Window size for computing acceleration.

        Returns:
            List of acceleration values per generation.
        """
        rates = self.get_growth_rate(window)
        if len(rates) < 2:
            return [0.0] * len(rates)

        acceleration = [0.0]  # First value has no previous rate
        for i in range(1, len(rates)):
            acceleration.append(rates[i] - rates[i-1])

        return acceleration

    def get_bottleneck_generations(self, threshold: float = 0.001,
                                    window: int = 5) -> List[int]:
        """Identify generations where improvement stalled.

        Args:
            threshold: Minimum improvement to consider as progress.
            window: Window size for computing average improvement.

        Returns:
            List of generation indices where improvement stalled.
        """
        best = self.history.get("best", [])
        bottlenecks = []

        for i in range(window, len(best)):
            recent_improvement = best[i] - np.mean(best[max(0, i-window):i])
            if abs(recent_improvement) < threshold:
                bottlenecks.append(i)

        return bottlenecks

    def get_quantiles(self, metric: str = "best",
                      quantiles: List[float] = None) -> Dict[float, float]:
        """Compute quantiles of a metric.

        Args:
            metric: Which metric to analyze.
            quantiles: List of quantiles to compute (0.0 to 1.0).

        Returns:
            Dictionary mapping quantile to value.
        """
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        values = self.history.get(metric, [])
        if not values:
            return {q: 0.0 for q in quantiles}

        result = {}
        for q in quantiles:
            result[q] = float(np.percentile(values, q * 100))

        return result

    def get_performance_bands(self, metric: str = "best",
                               n_bands: int = 5) -> List[Tuple[float, float, float]]:
        """Compute performance bands (percentile ranges).

        Args:
            metric: Which metric to analyze.
            n_bands: Number of bands to create.

        Returns:
            List of (lower_percentile, upper_percentile, (min_val, max_val)) tuples.
        """
        values = self.history.get(metric, [])
        if not values:
            return []

        bands = []
        band_size = 100 / n_bands

        for i in range(n_bands):
            lower_q = i * band_size
            upper_q = (i + 1) * band_size
            lower_val = float(np.percentile(values, lower_q))
            upper_val = float(np.percentile(values, upper_q))
            bands.append((lower_q / 100, upper_q / 100, (lower_val, upper_val)))

        return bands

    def compute_statistical_significance(self, run1_values: List[float],
                                          run2_values: List[float],
                                          alpha: float = 0.05) -> Dict[str, float]:
        """Compute statistical significance between two runs.

        Uses Welch's t-test approximation.

        Args:
            run1_values: Values from first run.
            run2_values: Values from second run.
            alpha: Significance level.

        Returns:
            Dictionary with t-statistic, p-value, and significance.
        """
        n1, n2 = len(run1_values), len(run2_values)
        if n1 < 2 or n2 < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False, "note": "insufficient_data"}

        mean1, mean2 = np.mean(run1_values), np.mean(run2_values)
        var1, var2 = np.var(run1_values, ddof=1), np.var(run2_values, ddof=1)

        # Welch's t-test
        se = np.sqrt(var1/n1 + var2/n2)
        if se == 0:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False, "note": "zero_variance"}

        t_stat = (mean1 - mean2) / se

        # Approximate p-value (two-tailed)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        # Simple approximation using normal distribution for large df
        p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": float(p_value) < alpha,
            "degrees_of_freedom": float(df),
            "mean_difference": float(mean1 - mean2),
        }

    def get_ema_curves(self, metric: str = "best",
                        alpha: float = 0.1) -> List[float]:
        """Compute exponential moving average of a metric.

        Args:
            metric: Which metric to smooth.
            alpha: Smoothing factor (0 < alpha <= 1).

        Returns:
            List of EMA values.
        """
        values = self.history.get(metric, [])
        if not values:
            return []

        ema = [values[0]]
        for i in range(1, len(values)):
            ema.append(alpha * values[i] + (1 - alpha) * ema[-1])

        return ema

    def get_generation_efficiency(self) -> Dict[str, float]:
        """Compute efficiency metrics (improvement per generation).

        Returns:
            Dictionary with efficiency statistics.
        """
        best = self.history.get("best", [])
        if len(best) < 2:
            return {"avg_improvement": 0.0, "max_improvement": 0.0,
                    "min_improvement": 0.0, "improvement_rate": 0.0}

        improvements = [best[i] - best[i-1] for i in range(1, len(best))]

        return {
            "avg_improvement": float(np.mean(improvements)),
            "max_improvement": float(max(improvements)),
            "min_improvement": float(min(improvements)),
            "improvement_rate": float(improvements[-1]) if improvements else 0.0,
            "total_improvement": float(best[-1] - best[0]),
            "generations_with_improvement": sum(1 for i in improvements if i > 0),
            "improvement_percentage": sum(1 for i in improvements if i > 0) / len(improvements) * 100,
        }


def _normal_cdf(x: float) -> float:
    """Approximate the cumulative distribution function of the standard normal distribution."""
    # Using a polynomial approximation (Abramowitz and Stegun)
    # This avoids numpy.erf which was removed in NumPy 2.0
    if x < -10:
        return 0.0
    if x > 10:
        return 1.0
    
    # Constants for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x_abs = abs(x) / np.sqrt(2)
    
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs * x_abs)
    
    return 0.5 * (1.0 + sign * y)
