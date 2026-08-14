"""Unit tests for monitoring, alerting, registry, and data pipeline modules."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetricsCollector:
    def test_init_default(self):
        from src.train.monitoring import MetricsCollector, MonitoringConfig
        collector = MetricsCollector()
        assert collector.config.enabled is True
        assert collector.running is False

    def test_record_generation(self):
        from src.train.monitoring import MetricsCollector
        collector = MetricsCollector()
        collector._start_time = time.time() - 100
        metrics = collector.record_generation(1, [1.0, 2.0, 3.0], diversity=0.5)
        assert metrics["best_fitness"] == 3.0
        assert metrics["mean_fitness"] == 2.0
        assert metrics["generation"] == 1

    def test_record_generation_empty(self):
        from src.train.monitoring import MetricsCollector
        collector = MetricsCollector()
        metrics = collector.record_generation(1, [])
        assert metrics == {}

    def test_get_fitness_curves(self):
        from src.train.monitoring import MetricsCollector
        collector = MetricsCollector()
        collector._start_time = time.time() - 100
        for i in range(5):
            collector.record_generation(i + 1, [float(i), float(i + 1)], diversity=float(i * 0.1))
        curves = collector.get_fitness_curves()
        assert len(curves["best"]) == 5
        assert len(curves["mean"]) == 5

    def test_convergence_detection(self):
        from src.train.monitoring import MetricsCollector
        collector = MetricsCollector()
        collector._start_time = time.time() - 100
        for i in range(15):
            collector.record_generation(i + 1, [10.0 + i * 0.001], diversity=0.1)
        convergence = collector._compute_convergence_indicator()
        assert convergence["converged"] is True

    def test_convergence_not_yet(self):
        from src.train.monitoring import MetricsCollector
        collector = MetricsCollector()
        collector._start_time = time.time() - 100
        for i in range(5):
            collector.record_generation(i + 1, [float(i)], diversity=float(i * 0.1))
        convergence = collector._compute_convergence_indicator()
        assert convergence["trend"] == "unknown"


class TestBottleneckDetector:
    def test_no_bottleneck(self):
        from src.train.monitoring import BottleneckDetector, BottleneckType
        detector = BottleneckDetector()
        reports = detector.detect(gpu_compute_pct=50, gpu_memory_pct=50, cpu_percent=50, gen_time_sec=60)
        assert any(r.bottleneck_type == BottleneckType.NONE for r in reports)

    def test_gpu_bound(self):
        from src.train.monitoring import BottleneckDetector, BottleneckType
        detector = BottleneckDetector()
        reports = detector.detect(gpu_compute_pct=98, gpu_memory_pct=92, cpu_percent=30, gen_time_sec=120)
        assert any(r.bottleneck_type == BottleneckType.GPU_BOUND for r in reports)
        assert any(r.severity == "high" for r in reports)

    def test_cpu_bound(self):
        from src.train.monitoring import BottleneckDetector, BottleneckType
        detector = BottleneckDetector()
        reports = detector.detect(gpu_compute_pct=30, gpu_memory_pct=30, cpu_percent=95, gen_time_sec=60)
        assert any(r.bottleneck_type == BottleneckType.CPU_BOUND for r in reports)

    def test_memory_bound(self):
        from src.train.monitoring import BottleneckDetector, BottleneckType
        detector = BottleneckDetector()
        reports = detector.detect(gpu_compute_pct=50, gpu_memory_pct=96, cpu_percent=50, gen_time_sec=60)
        assert any(r.bottleneck_type == BottleneckType.MEMORY_BOUND for r in reports)
        assert any(r.severity == "critical" for r in reports)

    def test_get_trend(self):
        from src.train.monitoring import BottleneckDetector
        detector = BottleneckDetector()
        for _ in range(15):
            detector.detect(gpu_compute_pct=50, gpu_memory_pct=50, cpu_percent=50, gen_time_sec=60)
        trend = detector.get_trend(window=10)
        assert "dominant_bottleneck" in trend
        assert "avg_gen_time" in trend


class TestAlertManager:
    def test_init(self):
        from src.alerting import AlertManager
        manager = AlertManager()
        assert len(manager.channels) >= 2

    def test_get_default_rules(self):
        from src.alerting import AlertManager
        manager = AlertManager()
        rules = manager.get_default_rules()
        assert len(rules) > 0

    def test_check_no_trigger(self):
        from src.alerting import AlertManager
        manager = AlertManager()
        alerts = manager.check({"converged": False, "early_stop": False, "training_complete": False})
        assert len(alerts) == 0

    def test_check_convergence(self):
        from src.alerting import AlertManager, AlertLevel, AlertRule, AlertType
        manager = AlertManager()
        manager.add_rule(AlertRule(AlertType.CONVERGENCE, AlertLevel.INFO, lambda ctx: ctx.get("converged", False), "Converged at gen {{{generation}}}", cooldown_sec=0))
        alerts = manager.check({"converged": True, "generation": 50, "best_fitness": 123.45})
        assert len(alerts) > 0
        assert any(a.level == AlertLevel.INFO for a in alerts)

    def test_check_early_stop(self):
        from src.alerting import AlertManager, AlertLevel, AlertRule, AlertType
        manager = AlertManager()
        manager.add_rule(AlertRule(AlertType.EARLY_STOP, AlertLevel.WARNING, lambda ctx: ctx.get("early_stop", False), "Early stop at gen {{{generation}}}", cooldown_sec=0))
        alerts = manager.check({"early_stop": True, "generation": 100, "patience": 30})
        assert len(alerts) > 0
        assert any(a.level == AlertLevel.WARNING for a in alerts)

    def test_check_training_complete(self):
        from src.alerting import AlertManager, AlertLevel, AlertRule, AlertType
        manager = AlertManager()
        manager.add_rule(AlertRule(AlertType.TRAINING_COMPLETE, AlertLevel.INFO, lambda ctx: ctx.get("training_complete", False), "Training complete: best {{{best_fitness}}}", cooldown_sec=0))
        alerts = manager.check({"training_complete": True, "best_fitness": 150.0, "total_gens": 100, "elapsed_min": 30.0})
        assert len(alerts) > 0
        assert any(a.level == AlertLevel.INFO for a in alerts)


class TestModelRegistry:
    def test_init(self, tmp_path):
        from src.registry import ModelRegistry
        registry = ModelRegistry(str(tmp_path / "registry"))
        assert len(registry.models) == 0

    def test_register_model(self, tmp_path):
        from src.registry import ModelRegistry, ModelStage
        registry = ModelRegistry(str(tmp_path / "registry"))
        model = registry.register_model(model_id="test_001", version="v1", fitness=123.45, architecture="cnn_lstm", param_count=50000, tags=["test"])
        assert model.model_id == "test_001"
        assert model.stage == ModelStage.CHECKPOINT
        assert model.fitness == 123.45
        assert len(registry.models) == 1

    def test_promote_model(self, tmp_path):
        from src.registry import ModelRegistry, ModelStage
        registry = ModelRegistry(str(tmp_path / "registry"))
        registry.register_model("test_001", "v1", fitness=100.0)
        model = registry.promote_model("test_001", ModelStage.PRODUCTION)
        assert model.stage == ModelStage.PRODUCTION

    def test_get_best_models(self, tmp_path):
        from src.registry import ModelRegistry
        registry = ModelRegistry(str(tmp_path / "registry"))
        registry.register_model("a", "v1", fitness=100.0)
        registry.register_model("b", "v1", fitness=200.0)
        registry.register_model("c", "v1", fitness=150.0)
        best = registry.get_best_models(limit=2)
        assert len(best) == 2
        assert best[0].fitness >= best[1].fitness

    def test_compare_models(self, tmp_path):
        from src.registry import ModelRegistry
        registry = ModelRegistry(str(tmp_path / "registry"))
        registry.register_model("a", "v1", fitness=100.0, architecture="cnn_lstm")
        registry.register_model("b", "v1", fitness=200.0, architecture="cnn_mlp")
        comparison = registry.compare_models(["a", "b"])
        assert "a" in comparison
        assert "b" in comparison
        assert comparison["a"]["fitness"] == 100.0

    def test_get_production_models(self, tmp_path):
        from src.registry import ModelRegistry, ModelStage
        registry = ModelRegistry(str(tmp_path / "registry"))
        registry.register_model("a", "v1", fitness=100.0)
        registry.register_model("b", "v1", fitness=200.0)
        registry.promote_model("a", ModelStage.PRODUCTION)
        prod = registry.get_production_models()
        assert len(prod) == 1
        assert prod[0].model_id == "a"


class TestMatchCollector:
    def test_collect_match(self, tmp_path):
        from src.train.data_pipeline import MatchCollector
        collector = MatchCollector()
        collector.output_dir = tmp_path / "matches"
        match = collector.collect_match(generation=1, agent1_id="a1", agent2_id="a2", state_snapshots=[{"tick": 0}], action_history=[{"action": 1}], reward_history=[{"reward": 0.5}], winner="a1")
        assert match.match_id == "gen0001_agent1_a1_agent2_a2"
        assert match.winner == "a1"

    def test_get_match_count(self, tmp_path):
        from src.train.data_pipeline import MatchCollector
        collector = MatchCollector()
        collector.output_dir = tmp_path / "matches"
        collector.collect_match(1, "a1", "a2", [], [], [], winner="a1")
        collector.collect_match(1, "a3", "a4", [], [], [], winner="a3")
        assert collector.get_match_count(1) == 2


class TestTrainingDataset:
    def test_create_dataset(self, tmp_path):
        from src.train.data_pipeline import TrainingDataset, DatasetSplit
        ds = TrainingDataset(str(tmp_path / "datasets"))
        version = ds.create_dataset("test_v1", ["run_1"], DatasetSplit.TRAIN)
        assert version.dataset_id == "test_v1"
        assert version.split == DatasetSplit.TRAIN

    def test_list_datasets(self, tmp_path):
        from src.train.data_pipeline import TrainingDataset, DatasetSplit
        ds = TrainingDataset(str(tmp_path / "datasets"))
        ds.create_dataset("v1", ["run_1"], DatasetSplit.TRAIN)
        ds.create_dataset("v2", ["run_2"], DatasetSplit.VAL)
        datasets = ds.list_datasets()
        assert len(datasets) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
