"""Tests for run management and advanced metrics.

Tests:
- RunMetadata
- RunManager (discovery, loading, comparison)
- Advanced metrics (growth rate, acceleration, quantiles, etc.)
- Statistical significance testing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.viz.metrics import TrainingMetrics
from src.viz.runs_manager import RunMetadata, RunManager


# =============================================================================
# RunMetadata Tests
# =============================================================================


class TestRunMetadata:
    """Tests for RunMetadata."""

    def test_is_complete(self):
        """Test is_complete property."""
        metadata = RunMetadata(end_time=100.0)
        assert metadata.is_complete is True

        metadata_no_end = RunMetadata(end_time=None)
        assert metadata_no_end.is_complete is False

    def test_progress(self):
        """Test progress calculation."""
        metadata = RunMetadata(max_generations=100, actual_generations=50)
        assert metadata.progress == 0.5

        metadata_complete = RunMetadata(max_generations=100, actual_generations=100)
        assert metadata_complete.progress == 1.0

        metadata_no_max = RunMetadata(actual_generations=50)
        assert metadata_no_max.progress == 0.0

    def test_to_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = RunMetadata(
            run_id="test_run",
            name="Test Run",
            description="A test run",
            start_time=1000.0,
            end_time=2000.0,
            duration_seconds=1000.0,
            max_generations=100,
            actual_generations=50,
            best_fitness=10.5,
            tags=["test", "experiment"],
        )

        data = original.to_dict()
        restored = RunMetadata.from_dict(data)

        assert restored.run_id == original.run_id
        assert restored.name == original.name
        assert restored.best_fitness == original.best_fitness
        assert restored.tags == original.tags


# =============================================================================
# RunManager Tests
# =============================================================================


class TestRunManager:
    """Tests for RunManager."""

    def test_get_summary_empty(self):
        """Test summary with no runs."""
        manager = RunManager(runs_dir="/nonexistent/path")
        summary = manager.get_summary()
        assert summary["total_runs"] == 0
        assert summary["completed_runs"] == 0
        assert summary["running_runs"] == 0

    def test_filter_by_tag(self):
        """Test filtering runs by tag."""
        runs = [
            RunMetadata(run_id="run1", tags=["test", "experiment"]),
            RunMetadata(run_id="run2", tags=["production"]),
            RunMetadata(run_id="run3", tags=["test", "dev"]),
        ]

        manager = RunManager()
        manager._runs = runs

        tagged = manager.filter_by_tag("test")
        assert len(tagged) == 2
        assert tagged[0].run_id in ["run1", "run3"]
        assert tagged[1].run_id in ["run1", "run3"]

    def test_get_run_tags(self):
        """Test getting all tags."""
        runs = [
            RunMetadata(run_id="run1", tags=["test", "experiment"]),
            RunMetadata(run_id="run2", tags=["production", "test"]),
        ]

        manager = RunManager()
        manager._runs = runs

        tags = manager.get_run_tags()
        assert "test" in tags
        assert "experiment" in tags
        assert "production" in tags
        assert "run1" in tags["test"]
        assert "run2" in tags["test"]


# =============================================================================
# Advanced Metrics Tests
# =============================================================================


class TestAdvancedMetrics:
    """Tests for advanced metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = TrainingMetrics()
        # Simulate 100 generations of fitness data
        np.random.seed(42)
        fitnesses = [1.0 + i * 0.1 + np.random.randn() * 0.05 for i in range(100)]
        self.metrics.history = {
            "best": fitnesses,
            "mean": [f + np.random.randn() * 0.1 for f in fitnesses],
            "median": [f + np.random.randn() * 0.08 for f in fitnesses],
            "min": [f - 1.0 for f in fitnesses],
            "max": [f + 1.0 for f in fitnesses],
            "std": [abs(np.random.randn()) * 0.5 for _ in fitnesses],
            "diversity": [10.0 + i * 0.05 + np.random.randn() * 0.5 for i in range(100)],
        }

    def test_growth_rate(self):
        """Test growth rate computation."""
        rates = self.metrics.get_growth_rate(window=10)
        assert len(rates) == 100
        # First 10 should be zero
        assert all(r == 0.0 for r in rates[:10])
        # Later should show mostly positive growth (fitness is increasing)
        positive_count = sum(1 for r in rates[10:] if r >= 0)
        assert positive_count > len(rates) * 0.5  # At least half should be non-negative

    def test_acceleration(self):
        """Test acceleration computation."""
        accel = self.metrics.get_acceleration(window=10)
        assert len(accel) == 100
        assert accel[0] == 0.0  # First value is always zero

    def test_bottleneck_detection(self):
        """Test bottleneck generation detection."""
        # Create data with a clear bottleneck
        metrics = TrainingMetrics()
        # First 50 gens: improving
        # Next 20 gens: flat (bottleneck)
        # Last 30 gens: improving again
        values = []
        for i in range(100):
            if i < 50:
                values.append(1.0 + i * 0.1)
            elif i < 70:
                values.append(6.0)  # Flat
            else:
                values.append(6.0 + (i - 70) * 0.1)

        metrics.history = {"best": values}
        bottlenecks = metrics.get_bottleneck_generations(threshold=0.01, window=3)

        # Should detect bottlenecks around generations 55-70
        assert len(bottlenecks) > 0
        # Most bottlenecks should be in the flat region
        in_flat_region = sum(1 for b in bottlenecks if 50 <= b <= 75)
        assert in_flat_region > len(bottlenecks) * 0.5

    def test_quantiles(self):
        """Test quantile computation."""
        quantiles = self.metrics.get_quantiles("best")
        assert 0.1 in quantiles
        assert 0.5 in quantiles
        assert 0.9 in quantiles
        # Median should be between 10th and 90th percentile
        assert quantiles[0.1] <= quantiles[0.5] <= quantiles[0.9]

    def test_performance_bands(self):
        """Test performance band computation."""
        bands = self.metrics.get_performance_bands("best", n_bands=5)
        assert len(bands) == 5
        # Bands should cover the full range
        assert bands[0][0] == 0.0  # Lower percentile of first band
        assert bands[-1][1] == 1.0  # Upper percentile of last band
        # Each band should have valid min/max
        for lo, hi, (min_val, max_val) in bands:
            assert lo < hi
            assert min_val <= max_val

    def test_statistical_significance_same(self):
        """Test significance test with similar runs."""
        metrics = TrainingMetrics()
        # Two very similar runs (low variance)
        np.random.seed(42)
        run1 = [10.0 + np.random.randn() * 0.01 for _ in range(100)]
        run2 = [10.0 + np.random.randn() * 0.01 for _ in range(100)]

        result = metrics.compute_statistical_significance(run1, run2)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "mean_difference" in result
        # Similar runs should not be significantly different
        assert result["significant"] is False

    def test_statistical_significance_different(self):
        """Test significance test with different runs."""
        metrics = TrainingMetrics()
        # Two clearly different runs (low variance for clear separation)
        np.random.seed(42)
        run1 = [10.0 + np.random.randn() * 0.01 for _ in range(100)]
        run2 = [15.0 + np.random.randn() * 0.01 for _ in range(100)]

        result = metrics.compute_statistical_significance(run1, run2, alpha=0.05)
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["mean_difference"] < 0  # run1 < run2

    def test_ema_curves(self):
        """Test exponential moving average."""
        ema = self.metrics.get_ema_curves("best", alpha=0.1)
        assert len(ema) == 100
        assert ema[0] == self.metrics.history["best"][0]  # First value same
        # EMA should be smoother than raw data
        raw_std = np.std(np.diff(self.metrics.history["best"]))
        ema_std = np.std(np.diff(ema))
        assert ema_std <= raw_std  # EMA should have less variance

    def test_generation_efficiency(self):
        """Test generation efficiency metrics."""
        efficiency = self.metrics.get_generation_efficiency()
        assert "avg_improvement" in efficiency
        assert "max_improvement" in efficiency
        assert "min_improvement" in efficiency
        assert "total_improvement" in efficiency
        assert "improvement_percentage" in efficiency
        assert 0 <= efficiency["improvement_percentage"] <= 100

    def test_convergence_status(self):
        """Test convergence detection."""
        # Non-converged run
        metrics = TrainingMetrics()
        metrics.history = {"best": list(range(50))}
        status = metrics.get_convergence_status(window=20)
        assert status["converged"] is False

        # Converged run (very stable values)
        metrics2 = TrainingMetrics()
        np.random.seed(123)
        metrics2.history = {"best": [10.0 + np.random.randn() * 0.00001 for _ in range(50)]}
        status2 = metrics2.get_convergence_status(window=20)
        assert status2["converged"] is True

    def test_smoothed_fitness(self):
        """Test fitness smoothing."""
        smoothed = self.metrics.get_smoothed_fitness(window=10, metric="best")
        assert len(smoothed) == 100
        # Smoothed should follow the original
        assert smoothed[-1] > smoothed[0]  # Overall trend preserved

    def test_pareto_front_single_objective(self):
        """Test Pareto front with single objective."""
        fitnesses = [1.0, 5.0, 3.0, 8.0, 2.0]
        pareto = self.metrics.get_pareto_front(fitnesses)
        # Should include the best value
        assert 3 in pareto  # Index of max value (8.0)

    def test_pareto_front_multi_objective(self):
        """Test Pareto front with multi-objective."""
        fitnesses = [1.0, 5.0, 3.0, 4.0, 2.0]
        diversity = [1.0, 1.0, 5.0, 4.0, 5.0]
        pareto = self.metrics.get_pareto_front(fitnesses, diversity)
        # Should include non-dominated solutions
        assert len(pareto) > 0

    def test_summary(self):
        """Test metrics summary."""
        summary = self.metrics.get_summary()
        assert "total_generations" in summary
        assert "best_fitness" in summary
        assert "best_generation" in summary
        assert "convergence" in summary
        assert summary["total_generations"] == 100

    def test_add_from_history(self):
        """Test adding data from existing history."""
        metrics = TrainingMetrics()
        history = {
            "best": [1.0, 2.0, 3.0],
            "mean": [0.9, 1.9, 2.9],
        }
        metrics.add_from_history(history)
        assert len(metrics.history["best"]) == 3
        assert len(metrics.history["mean"]) == 3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_metrics(self):
        """Test with empty metrics."""
        metrics = TrainingMetrics()
        assert metrics.get_growth_rate() == []
        assert metrics.get_acceleration() == []
        assert metrics.get_quantiles() == {0.1: 0.0, 0.25: 0.0, 0.5: 0.0, 0.75: 0.0, 0.9: 0.0}
        assert metrics.get_performance_bands() == []
        assert metrics.get_generation_efficiency() == {"avg_improvement": 0.0, "max_improvement": 0.0,
                                                        "min_improvement": 0.0, "improvement_rate": 0.0}

    def test_single_generation(self):
        """Test with single generation."""
        metrics = TrainingMetrics()
        metrics.history = {"best": [5.0]}
        assert metrics.get_growth_rate() == [0.0]
        assert metrics.get_acceleration() == [0.0]
        efficiency = metrics.get_generation_efficiency()
        assert efficiency["avg_improvement"] == 0.0

    def test_statistical_test_insufficient_data(self):
        """Test statistical significance with insufficient data."""
        metrics = TrainingMetrics()
        result = metrics.compute_statistical_significance([1.0], [2.0])
        assert result["significant"] is False
        assert result.get("note") == "insufficient_data"

    def test_run_metadata_default_values(self):
        """Test RunMetadata default values."""
        metadata = RunMetadata()
        assert metadata.run_id == ""
        assert metadata.name == "Unnamed Run"
        assert metadata.description == ""
        assert metadata.start_time is None
        assert metadata.end_time is None
        assert metadata.duration_seconds == 0.0
        assert metadata.max_generations == 0
        assert metadata.actual_generations == 0
        assert metadata.best_fitness == 0.0
        assert metadata.tags == []
