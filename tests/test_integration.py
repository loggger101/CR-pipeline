"""Integration tests for the CR-Pipeline framework.

Tests end-to-end workflows including:
- Training pipeline execution
- Tournament evaluation
- Hyperparameter optimization
- Architecture search
- Model export
- Experiment tracking
- Report generation
- Pipeline orchestration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest


# =============================================================================
# Training Pipeline Integration
# =============================================================================


class TestTrainingPipeline:
    """Integration tests for training pipeline."""

    def test_evolution_trainer_initialization(self):
        """Test EvolutionTrainer can be initialized."""
        from src.train import TrainingConfig

        config = TrainingConfig(
            population_size=50,
            elite_count=5,
            max_generations=10,
            num_workers=1,
            tournament_mode=False,
        )

        assert config.population_size == 50
        assert config.max_generations == 10
        assert config.tournament_mode is False

    def test_tournament_config(self):
        """Test tournament mode configuration."""
        from src.train import TrainingConfig

        config = TrainingConfig(
            population_size=50,
            elite_count=5,
            max_generations=10,
            tournament_mode=True,
            tournament_format="round_robin",
            tournament_matches=2,
            tournament_elite_fraction=0.1,
        )

        assert config.tournament_mode is True
        assert config.tournament_format == "round_robin"
        assert config.tournament_matches == 2

    def test_config_serialization(self):
        """Test TrainingConfig to_dict/from_dict."""
        from src.train import TrainingConfig

        config = TrainingConfig(
            population_size=100,
            elite_count=10,
            max_generations=50,
            crossover_rate=0.8,
            mutation_rate=0.05,
        )

        data = config.to_dict()
        restored = TrainingConfig.from_dict(data)

        assert restored.population_size == config.population_size
        assert restored.max_generations == config.max_generations
        assert restored.crossover_rate == config.crossover_rate


# =============================================================================
# Tournament System Integration
# =============================================================================


class TestTournamentIntegration:
    """Integration tests for tournament system."""

    def test_tournament_result_creation(self):
        """Test creating and using TournamentResult."""
        from src.train import (
            TournamentResult,
            AgentTournamentStats,
            HeadToHeadRecord,
        )

        stats_a = AgentTournamentStats(agent_id="a", wins=5, draws=1, losses=4)
        stats_b = AgentTournamentStats(agent_id="b", wins=4, draws=2, losses=4)

        h2h = HeadToHeadRecord(
            agent1_id="a", agent2_id="b",
            agent1_wins=3, agent2_wins=2, draws=2,
        )

        result = TournamentResult(
            rankings=[("a", 7.5), ("b", 6.0)],
            agent_stats={"a": stats_a, "b": stats_b},
            h2h_records={("a", "b"): h2h},
            total_matches=10,
            elo_ratings={"a": 1550.0, "b": 1480.0},
            generation=5,
        )

        assert result.get_ranking("a") == 1
        assert result.get_ranking("b") == 2
        assert result.get_win_rate("a") == pytest.approx(5 / 10)
        assert result.get_h2h_record("a", "b") is h2h

    def test_tournament_format_enum(self):
        """Test TournamentFormat enum values."""
        from src.train import TournamentFormat

        assert TournamentFormat.ROUND_ROBIN is not None
        assert TournamentFormat.SINGLE_ELIMINATION is not None
        assert TournamentFormat.DOUBLE_ELIMINATION is not None
        assert TournamentFormat.LEAGUE is not None


# =============================================================================
# Hyperparameter Optimization Integration
# =============================================================================


class TestHPOIntegration:
    """Integration tests for hyperparameter optimization."""

    def test_param_space_sampling(self):
        """Test ParamSpace sampling."""
        from src.train import ParamSpace, ParamType

        int_param = ParamSpace("test_int", ParamType.INTEGER, low=1, high=10)
        float_param = ParamSpace("test_float", ParamType.FLOAT, low=0.0, high=1.0)
        log_param = ParamSpace("test_log", ParamType.LOG_FLOAT, low=0.001, high=0.1)
        cat_param = ParamSpace("test_cat", ParamType.CATEGORICAL, values=["a", "b", "c"])

        import numpy as np
        rng = np.random.RandomState(42)

        # Test integer sampling
        for _ in range(10):
            val = int_param.sample(rng)
            assert 1 <= val <= 10

        # Test float sampling
        for _ in range(10):
            val = float_param.sample(rng)
            assert 0.0 <= val <= 1.0

        # Test log float sampling
        for _ in range(10):
            val = log_param.sample(rng)
            assert 0.001 <= val <= 0.1

        # Test categorical sampling
        for _ in range(10):
            val = cat_param.sample(rng)
            assert val in ["a", "b", "c"]

    def test_bayesian_optimizer_structure(self):
        """Test BayesianOptimizer initialization."""
        from src.train import BayesianOptimizer, ParamSpace, ParamType

        param_spaces = [
            ParamSpace("x", ParamType.FLOAT, low=0.0, high=1.0),
            ParamSpace("y", ParamType.INTEGER, low=1, high=10),
        ]

        optimizer = BayesianOptimizer(
            param_spaces=param_spaces,
            n_initial=5,
            n_iterations=10,
            seed=42,
        )

        assert optimizer.n_initial == 5
        assert optimizer.n_iterations == 10

    def test_grid_search_structure(self):
        """Test GridSearchOptimizer initialization."""
        from src.train import GridSearchOptimizer, ParamSpace, ParamType

        param_spaces = [
            ParamSpace("x", ParamType.FLOAT, low=0.0, high=1.0),
            ParamSpace("y", ParamType.INTEGER, low=1, high=3),
        ]

        optimizer = GridSearchOptimizer(param_spaces, grid_points=3)

        # Should have 3 * 3 = 9 configurations
        assert len(optimizer._grid) == 9

    def test_random_search_structure(self):
        """Test RandomSearchOptimizer initialization."""
        from src.train import RandomSearchOptimizer, ParamSpace, ParamType

        param_spaces = [
            ParamSpace("x", ParamType.FLOAT, low=0.0, high=1.0),
        ]

        optimizer = RandomSearchOptimizer(
            param_spaces=param_spaces,
            n_trials=20,
            patience=5,
            seed=42,
        )

        assert optimizer.n_trials == 20
        assert optimizer.patience == 5

    def test_sensitivity_analyzer(self):
        """Test sensitivity analysis."""
        from src.train import SensitivityAnalyzer, ParamSpace, ParamType

        param_spaces = [
            ParamSpace("x", ParamType.FLOAT, low=0.0, high=1.0),
            ParamSpace("y", ParamType.FLOAT, low=0.0, high=1.0),
        ]

        analyzer = SensitivityAnalyzer(param_spaces)

        results = [
            {"params": {"x": 0.1, "y": 0.2}, "score": 1.0},
            {"params": {"x": 0.5, "y": 0.5}, "score": 2.0},
            {"params": {"x": 0.9, "y": 0.8}, "score": 3.0},
        ]

        sensitivities = analyzer.analyze(results)
        assert "x" in sensitivities
        assert "y" in sensitivities
        assert 0 <= sensitivities["x"] <= 1
        assert 0 <= sensitivities["y"] <= 1


# =============================================================================
# Experiment Tracking Integration
# =============================================================================


class TestExperimentTrackingIntegration:
    """Integration tests for experiment tracking."""

    def test_experiment_creation_and_run(self):
        """Test creating experiment and starting run."""
        from src.train import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)

            exp = tracker.create_experiment("Test Experiment", tags=["test"])
            assert exp.experiment_id is not None
            assert exp.name == "Test Experiment"
            assert "test" in exp.tags

            run = tracker.start_run(
                exp.experiment_id,
                name="Test Run",
                params={"population_size": 100},
                tags=["exp1"],
            )
            assert run.run_id is not None
            assert run.name == "Test Run"
            assert run.params.get("population_size") == 100

    def test_metric_logging(self):
        """Test logging metrics."""
        from src.train import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)

            exp = tracker.create_experiment("Metric Test")
            run = tracker.start_run(exp.experiment_id)

            # Log individual metrics
            tracker.log_metric(run.run_id, "fitness", 1.0, step=0)
            tracker.log_metric(run.run_id, "fitness", 2.0, step=1)
            tracker.log_metric(run.run_id, "fitness", 3.0, step=2)

            # Log batch metrics
            tracker.log_metrics_batch(run.run_id, {"diversity": 5.0, "speed": 100.0}, step=0)

            # Verify
            run_data = tracker.get_run(run.run_id)
            assert len(run_data.metrics["fitness"]) == 3
            assert run_data.metrics["fitness"][0].value == 1.0
            assert run_data.metrics["fitness"][2].value == 3.0
            assert "diversity" in run_data.metrics
            assert "speed" in run_data.metrics

    def test_run_end(self):
        """Test ending a run."""
        from src.train import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)

            exp = tracker.create_experiment("End Test")
            run = tracker.start_run(exp.experiment_id)

            tracker.end_run(run.run_id, status="completed")

            # Run should no longer be active
            assert run.run_id not in tracker.get_active_runs()
            assert run.status == "completed"

    def test_experiment_summary(self):
        """Test experiment summary generation."""
        from src.train import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)

            exp = tracker.create_experiment("Summary Test")
            for i in range(3):
                run = tracker.start_run(exp.experiment_id)
                tracker.log_metric(run.run_id, "fitness", 10.0 + i, step=0)
                tracker.end_run(run.run_id, status="completed")

            summary = exp.get_summary()
            assert summary["total_runs"] == 3
            assert summary["completed_runs"] == 3
            assert "fitness" in summary["metrics"]
            assert summary["metrics"]["fitness"]["best"] == 12.0

    def test_run_comparison(self):
        """Test run comparison."""
        from src.train import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)

            exp = tracker.create_experiment("Comparison Test")
            runs = []
            for i in range(3):
                run = tracker.start_run(exp.experiment_id)
                tracker.log_metric(run.run_id, "fitness", 10.0 + i * 2, step=0)
                tracker.end_run(run.run_id, status="completed")
                runs.append(run.run_id)

            comparison = tracker.compare_runs(runs, "fitness")
            assert comparison["best_run"] is not None
            assert comparison["best_value"] == 14.0
            assert comparison["spread"] == 4.0

    def test_report_generation(self):
        """Test report generation."""
        from src.train import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tracking_dir=tmpdir)

            exp = tracker.create_experiment("Report Test")
            run = tracker.start_run(exp.experiment_id)
            tracker.log_metric(run.run_id, "fitness", 15.5, step=0)
            tracker.log_param(run.run_id, "population_size", 200)
            tracker.end_run(run.run_id, status="completed")

            # Generate HTML report
            report_path = tracker.generate_report(exp.experiment_id, "fitness")
            assert report_path is not None

            # Generate Markdown report
            from src.viz import ReportGenerator
            reporter = ReportGenerator(output_dir=tmpdir)
            md_path = reporter.generate_markdown_report(
                {"name": exp.name, "best_fitness": 15.5, "generations": 100},
                "training",
            )
            assert md_path is not None

            # Generate JSON report
            json_path = reporter.generate_json_report(
                {"type": "test", "data": {"key": "value"}},
            )
            assert json_path is not None
            
            # Verify JSON is valid
            import json as json_module
            with open(json_path) as f:
                json_module.load(f)


# =============================================================================
# Architecture Search Integration
# =============================================================================


class TestArchitectureSearchIntegration:
    """Integration tests for architecture search."""

    def test_architecture_sampling(self):
        """Test random architecture sampling."""
        from src.models import ArchitectureSearchSpace

        arch = ArchitectureSearchSpace.sample()
        assert arch.name is not None
        assert len(arch.layers) > 0

    def test_architecture_evolution(self):
        """Test architecture evolution."""
        from src.models import ArchitectureEvolver, ArchitectureSearchSpace

        # Use a simplified search space
        custom_space = ArchitectureSearchSpace.get_default_space()

        evolver = ArchitectureEvolver(
            search_space=custom_space,
            population_size=5,
            elite_fraction=0.2,
            seed=42,
        )

        evolver.initialize_population()
        assert len(evolver.population) == 5

        # Simple fitness function
        def fitness_fn(arch):
            return len(arch.layers) * 0.1 + 0.01

        best_arch, info = evolver.evolve_generation(
            fitness_fn=fitness_fn,
            n_generations=3,
        )

        # best_arch might be a list
        if isinstance(best_arch, list):
            assert len(best_arch) > 0
        else:
            assert best_arch is not None
        assert "fitness_history" in info
        assert "diversity_history" in info

    def test_architecture_registry(self):
        """Test architecture registry."""
        from src.models import ArchitectureRegistry, ArchitectureConfig
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArchitectureRegistry(registry_dir=tmpdir)

            arch = ArchitectureConfig(
                name="test_arch",
                layers=[{"type": "conv2d", "filters": 32}],
            )

            registry.register_architecture("arch_1", arch, fitness=10.0)
            retrieved = registry.get_architecture("arch_1")
            assert retrieved is not None
            assert retrieved.name == "test_arch"

            summary = registry.get_architecture_summary()
            assert summary["total_architectures"] == 1


# =============================================================================
# Ensemble Methods Integration
# =============================================================================


class TestEnsembleIntegration:
    """Integration tests for ensemble methods."""

    def test_weight_averaging_ensemble(self):
        """Test weight averaging ensemble."""
        from src.models import WeightAveragingEnsemble

        models = [np.ones(100) * i for i in range(1, 4)]
        fitnesses = [10.0, 8.0, 6.0]

        ensemble = WeightAveragingEnsemble(use_performance_weighting=True)
        combined = ensemble.combine(models, fitnesses=fitnesses)

        assert combined.shape == models[0].shape
        # Should be weighted average favoring higher fitness
        assert np.all(combined > 0)

    def test_geometric_mean_ensemble(self):
        """Test geometric mean ensemble."""
        from src.models import GeometricMeanEnsemble

        models = [np.abs(np.random.randn(100)) + 0.1 for _ in range(3)]

        ensemble = GeometricMeanEnsemble()
        combined = ensemble.combine(models)

        assert combined.shape == models[0].shape

    def test_diversity_metric(self):
        """Test diversity metric computation."""
        from src.models import DiversityMetric

        # Create diverse models
        models = [np.random.randn(100) * i for i in range(1, 4)]

        # Pairwise distance
        distances = DiversityMetric.pairwise_distance(models)
        assert distances.shape == (3, 3)
        assert np.all(distances >= 0)
        assert np.all(distances[np.diag_indices_from(distances)] == 0)

    def test_ensemble_builder(self):
        """Test ensemble builder."""
        from src.models import EnsembleBuilder

        models = [np.random.randn(100) for _ in range(3)]
        fitnesses = [10.0, 8.0, 6.0]

        builder = EnsembleBuilder()
        result = builder.build_ensemble(
            models=models,
            fitnesses=fitnesses,
            method="weighted_average",
            optimize=True,
            n_iterations=10,
        )

        assert result.combined_weights.shape == models[0].shape
        assert result.ensemble_score > 0
        assert result.method == "weighted_average"


# =============================================================================
# Pipeline Integration
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline orchestration."""

    def test_evolution_pipeline_creation(self):
        """Test creating evolution pipeline."""
        from src.train import create_evolution_pipeline

        pipeline = create_evolution_pipeline(
            population_size=50,
            max_generations=10,
            use_tournament=False,
        )

        assert pipeline.name == "evolution"
        assert len(pipeline.stages) >= 4  # At least 4 stages

    def test_hpo_pipeline_creation(self):
        """Test creating HPO pipeline."""
        from src.train import create_hpo_pipeline

        pipeline = create_hpo_pipeline(
            optimizer_type="bayesian",
            n_trials=50,
        )

        assert "hpo_bayesian" in pipeline.name
        assert len(pipeline.stages) >= 3

    def test_pipeline_execution(self):
        """Test pipeline execution."""
        from src.train import create_evolution_pipeline

        pipeline = create_evolution_pipeline(
            population_size=50,
            max_generations=10,
            use_tournament=False,
        )

        # Dry run
        result = pipeline.run(dry_run=True)
        # Dry run returns empty dict but should not error

        # Actual run
        result = pipeline.run()
        # Pipeline stages execute and return results

        # Check status
        status = pipeline.get_status()
        assert status["total_stages"] > 0

    def test_pipeline_visualization(self):
        """Test pipeline ASCII visualization."""
        from src.train import create_evolution_pipeline

        pipeline = create_evolution_pipeline(
            population_size=50,
            max_generations=10,
        )

        viz = pipeline.get_visualization()
        assert "evolution" in viz.lower()
        assert "=" in viz

    def test_pipeline_checkpoint_resume(self):
        """Test pipeline checkpoint and resume."""
        from src.train import Pipeline, PipelineStage
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            def stage1_fn(context):
                return {"stage1_done": True}

            def stage2_fn(context):
                return {"stage2_done": True}

            pipeline = Pipeline(
                "test_pipeline",
                checkpoint_dir=tmpdir,
            )
            pipeline.add_stage(PipelineStage(name="stage1", fn=stage1_fn))
            pipeline.add_stage(PipelineStage(name="stage2", fn=stage2_fn, depends_on=["stage1"]))

            # Run
            result = pipeline.run()
            assert "stage1" in result
            assert "stage2" in result

            # Run again (should use checkpoint)
            result2 = pipeline.run()
            assert len(result2) == len(result)


# =============================================================================
# Report Generation Integration
# =============================================================================


class TestReportGenerationIntegration:
    """Integration tests for report generation."""

    def test_training_report_generation(self):
        """Test training report generation."""
        from src.viz import ReportGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)

            metrics = {
                "best_fitness": 15.5,
                "mean_fitness": 12.0,
                "diversity": 5.0,
                "actual_generations": 100,
            }
            fitness_history = {
                "best": [10.0 + i * 0.05 for i in range(100)],
                "mean": [8.0 + i * 0.04 for i in range(100)],
            }

            report_path = reporter.generate_training_report(
                run_dir="/tmp",
                output_filename="test_report.html",
            )
            assert report_path is not None

    def test_experiment_report_generation(self):
        """Test experiment report generation."""
        from src.viz import ReportGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)

            exp_data = {
                "name": "Test Experiment",
                "experiment_id": "exp_test",
                "summary": {
                    "total_runs": 10,
                    "completed_runs": 8,
                    "duration_hours": 2.5,
                },
            }

            report_path = reporter.generate_experiment_report(
                exp_data,
                "exp_report.html",
            )
            assert report_path is not None

    def test_tournament_report_generation(self):
        """Test tournament report generation."""
        from src.viz import ReportGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)

            tournament_data = {
                "rankings": [("agent_0", 10.0), ("agent_1", 8.0)],
                "elo_ratings": {"agent_0": 1550.0, "agent_1": 1480.0},
                "summary": {"competitiveness": 0.8},
            }

            report_path = reporter.generate_tournament_report(
                tournament_data,
                "tournament_report.html",
            )
            assert report_path is not None

    def test_comparison_report_generation(self):
        """Test comparison report generation."""
        from src.viz import ReportGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)

            runs_data = [
                {"name": "Run 1", "best_fitness": 10.0, "generations": 100, "duration": 60.0},
                {"name": "Run 2", "best_fitness": 12.0, "generations": 100, "duration": 65.0},
            ]

            report_path = reporter.generate_comparison_report(
                runs_data,
                "comparison_report.html",
            )
            assert report_path is not None

    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        from src.viz import ReportGenerator, ReportType
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)

            data = {"best_fitness": 15.5, "generations": 100}
            report_path = reporter.generate_markdown_report(
                data,
                ReportType.TRAINING,
                "test_report.md",
            )
            assert report_path is not None
            assert report_path.endswith(".md")

    def test_json_report_generation(self):
        """Test JSON report generation."""
        from src.viz import ReportGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = ReportGenerator(output_dir=tmpdir)

            data = {"type": "test", "data": {"key": "value"}}
            report_path = reporter.generate_json_report(
                data,
                "test_report.json",
            )
            assert report_path is not None

            # Verify JSON is valid
            with open(report_path) as f:
                json_data = json.load(f)
            # Report wraps data in a structure
            assert json_data.get("data", {}).get("type") == "test" or json_data.get("type") == "test"


# =============================================================================
# Model Export Integration
# =============================================================================


class TestModelExportIntegration:
    """Integration tests for model export."""

    def test_model_export_numpy(self):
        """Test NumPy model export."""
        from src.deploy import ModelExporter, ModelMetadata, ModelFormat
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=tmpdir)

            weights = np.random.randn(100, 50)
            metadata = ModelMetadata(
                model_id="test_model",
                version="1.0",
                format=ModelFormat.NUMPY,
                architecture="test",
                input_shape=[1, 50],
                output_shape=[1, 100],
            )

            export_path = exporter.export_model(weights, metadata, ModelFormat.NUMPY)
            assert export_path is not None
            assert export_path.endswith(".npy")

            # Verify file can be loaded
            loaded = np.load(export_path)
            assert loaded.shape == weights.shape

    def test_model_export_json(self):
        """Test JSON model export."""
        from src.deploy import ModelExporter, ModelMetadata, ModelFormat
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(output_dir=tmpdir)

            weights = np.random.randn(10, 5)
            metadata = ModelMetadata(
                model_id="test_model",
                version="1.0",
                format=ModelFormat.JSON,
                architecture="test",
                input_shape=[1, 5],
                output_shape=[1, 10],
            )

            export_path = exporter.export_model(weights, metadata, ModelFormat.JSON)
            assert export_path is not None
            assert export_path.endswith(".json")

            # Verify JSON is valid
            with open(export_path) as f:
                json_data = json.load(f)
            assert "weights" in json_data
            assert json_data["shape"] == list(weights.shape)

    def test_model_compression(self):
        """Test model compression."""
        from src.deploy import ModelCompressor

        compressor = ModelCompressor()

        weights = np.random.randn(100, 50)

        # Test pruning
        pruned, sparsity = compressor.prune_weights(weights, threshold=0.1)
        assert pruned.shape == weights.shape
        assert 0 <= sparsity <= 1

        # Test quantization
        quantized, ratio = compressor.quantize_weights(weights, bits=8)
        assert quantized.shape == weights.shape
        assert ratio == 4.0  # 32/8

    def test_inference_benchmarking(self):
        """Test inference benchmarking."""
        from src.deploy import InferenceBenchmarker

        benchmarker = InferenceBenchmarker()

        def model_fn(x):
            return x @ np.eye(x.shape[1]).T

        results = benchmarker.benchmark(
            model_fn=model_fn,
            input_shape=(1, 10),
            n_runs=10,
            warmup_runs=2,
        )

        assert "mean_latency_ms" in results
        assert "mean_throughput" in results
        assert results["n_runs"] == 10


# =============================================================================
# Full End-to-End Workflow
# =============================================================================


class TestFullWorkflow:
    """Integration tests for full workflow."""

    def test_complete_workflow(self):
        """Test complete training workflow."""
        from src.train import (
            ExperimentTracker,
            TrainingConfig,
            create_evolution_pipeline,
        )
        from src.viz import ReportGenerator
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create experiment
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            exp = tracker.create_experiment("Full Workflow Test", tags=["e2e", "test"])

            # 2. Start run
            config = TrainingConfig(
                population_size=50,
                elite_count=5,
                max_generations=5,
                num_workers=1,
            )
            run = tracker.start_run(
                exp.experiment_id,
                name="E2E Run",
                params=config.to_dict(),
            )

            # 3. Log metrics
            tracker.log_metrics_batch(run.run_id, {
                "best_fitness": 10.0,
                "mean_fitness": 8.0,
                "diversity": 5.0,
            }, step=0)

            # 4. End run
            tracker.end_run(run.run_id, status="completed")

            # 5. Create pipeline
            pipeline = create_evolution_pipeline(
                population_size=50,
                max_generations=5,
            )
            result = pipeline.run(dry_run=True)

            # 6. Generate report
            reporter = ReportGenerator(output_dir=tmpdir)
            report = tracker.generate_report(exp.experiment_id, "best_fitness")

            # Verify
            assert exp.experiment_id is not None
            assert run.run_id is not None
            assert report is not None

    def test_hpo_workflow(self):
        """Test hyperparameter optimization workflow."""
        from src.train import (
            BayesianOptimizer,
            ParamSpace,
            ParamType,
            ExperimentTracker,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment
            tracker = ExperimentTracker(tracking_dir=tmpdir)
            exp = tracker.create_experiment("HPO Test")

            # Define search space
            param_spaces = [
                ParamSpace("learning_rate", ParamType.LOG_FLOAT, 0.001, 0.1),
                ParamSpace("batch_size", ParamType.INTEGER, 16, 64),
            ]

            # Simple objective function
            def objective(params):
                return 10 - abs(params["learning_rate"] - 0.05) * 100

            # Run optimization
            optimizer = BayesianOptimizer(
                param_spaces=param_spaces,
                n_initial=3,
                n_iterations=5,
                seed=42,
            )
            result = optimizer.optimize(objective)

            # Log to experiment
            run = tracker.start_run(exp.experiment_id, name="HPO Run")
            tracker.log_metric(run.run_id, "best_score", result.best_score, step=0)
            tracker.end_run(run.run_id, status="completed")

            # Verify
            assert result.best_score > 0
            assert result.n_evaluations > 0


# =============================================================================
# Metrics Integration
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for metrics computation."""

    def test_advanced_metrics_computation(self):
        """Test advanced metrics computation."""
        from src.viz import TrainingMetrics
        import numpy as np

        metrics = TrainingMetrics()

        # Simulate 100 generations
        fitnesses = [1.0 + i * 0.1 + np.random.randn() * 0.05 for i in range(100)]
        diversity = [10.0 + i * 0.05 + np.random.randn() * 0.5 for i in range(100)]

        for i, f in enumerate(fitnesses):
            metrics.update([f], diversity[i], generation=i)

        # Test growth rate
        growth = metrics.get_growth_rate(window=10)
        assert len(growth) == 100

        # Test acceleration
        accel = metrics.get_acceleration(window=10)
        assert len(accel) == 100

        # Test quantiles
        quantiles = metrics.get_quantiles("best")
        assert 0.5 in quantiles
        assert quantiles[0.1] <= quantiles[0.5] <= quantiles[0.9]

        # Test performance bands
        bands = metrics.get_performance_bands("best", n_bands=5)
        assert len(bands) == 5

        # Test bottleneck detection
        bottlenecks = metrics.get_bottleneck_generations()
        assert isinstance(bottlenecks, list)

        # Test EMA
        ema = metrics.get_ema_curves("best", alpha=0.1)
        assert len(ema) == 100

        # Test efficiency
        efficiency = metrics.get_generation_efficiency()
        assert "avg_improvement" in efficiency
        assert "total_improvement" in efficiency

    def test_statistical_significance(self):
        """Test statistical significance computation."""
        from src.viz import TrainingMetrics
        import numpy as np

        metrics = TrainingMetrics()

        # Similar runs
        run1 = [10.0 + np.random.randn() * 0.1 for _ in range(50)]
        run2 = [10.0 + np.random.randn() * 0.1 for _ in range(50)]

        result = metrics.compute_statistical_significance(run1, run2)
        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result

        # Different runs
        run3 = [20.0 + np.random.randn() * 0.1 for _ in range(50)]
        result2 = metrics.compute_statistical_significance(run1, run3, alpha=0.05)
        assert result2["significant"] is True

    def test_convergence_detection(self):
        """Test convergence detection."""
        from src.viz import TrainingMetrics
        import numpy as np

        # Non-converged
        metrics1 = TrainingMetrics()
        metrics1.history = {"best": list(range(50))}
        status1 = metrics1.get_convergence_status(window=20)
        assert status1["converged"] is False

        # Converged
        metrics2 = TrainingMetrics()
        np.random.seed(123)
        metrics2.history = {"best": [10.0 + np.random.randn() * 0.00001 for _ in range(50)]}
        status2 = metrics2.get_convergence_status(window=20)
        assert status2["converged"] is True

    def test_pareto_front(self):
        """Test Pareto front computation."""
        from src.viz import TrainingMetrics

        metrics = TrainingMetrics()

        # Single objective
        fitnesses = [1.0, 5.0, 3.0, 8.0, 2.0]
        pareto = metrics.get_pareto_front(fitnesses)
        assert 3 in pareto  # Index of max value

        # Multi-objective
        fitnesses2 = [1.0, 5.0, 3.0, 4.0, 2.0]
        diversity = [1.0, 1.0, 5.0, 4.0, 5.0]
        pareto2 = metrics.get_pareto_front(fitnesses2, diversity)
        assert len(pareto2) > 0

    def test_summary_and_history(self):
        """Test metrics summary and history management."""
        from src.viz import TrainingMetrics

        metrics = TrainingMetrics()

        # Update with data
        for i in range(10):
            fitnesses = [1.0 + i + np.random.randn() * 0.1 for _ in range(50)]
            metrics.update(fitnesses, diversity=5.0 + i, generation=i)

        # Test summary
        summary = metrics.get_summary()
        assert summary["total_generations"] == 10
        assert summary["best_fitness"] > 0

        # Test fitness curves
        curves = metrics.get_fitness_curves()
        assert "best" in curves
        assert "mean" in curves
        assert len(curves["best"]) == 10

        # Test add from history
        new_metrics = TrainingMetrics()
        new_metrics.add_from_history(curves)
        assert len(new_metrics.history["best"]) == 10
