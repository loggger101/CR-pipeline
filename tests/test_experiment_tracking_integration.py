"""Tests that training runs register with the experiment tracker when enabled.

The tracker was fully implemented and unit-tested, but nothing in the
training loop ever wrote to it -- `crp experiments` could only see manually
created entries. These tests guard the wiring: a trainer run must appear as
an experiment with a completed (or stopped) run carrying per-generation
metrics, params and an artifact pointer back to the run directory.

Also covers the report generators reading real trainer artifacts in all three
formats (HTML/Markdown/JSON), which previously only worked for HTML -- and
only after key-name normalisation.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import EvolutionTrainer, TrainingConfig
from src.train.experiment_tracking import ExperimentTracker
from src.viz.reports import ReportGenerator


def _config(runs_dir: str, tracking_dir: str, **overrides) -> TrainingConfig:
    params = dict(
        population_size=6, elite_count=2, max_generations=2, num_workers=2,
        match_duration="short", runs_dir=str(runs_dir), seed=5,
        checkpoint_interval=100, curriculum_learning=False,
        diversity_preservation=False, tournament_matches=2,
        hall_of_fame_size=2, enable_experiment_tracking=True,
        experiment_tracking_dir=str(tracking_dir),
    )
    params.update(overrides)
    return TrainingConfig(**params)


class TestTrainerExperimentTracking:

    def test_run_is_registered_completed_and_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = os.path.join(tmp, "run")
            track = os.path.join(tmp, "track")
            config = _config(runs, track)
            with EvolutionTrainer(config) as trainer:
                trainer.train()

            # Re-open from disk to prove persistence, not just in-memory state.
            tracker = ExperimentTracker(track)
            experiments = [e for e in tracker.list_experiments() if "training" in e.tags]
            assert len(experiments) == 1
            exp = experiments[0]
            run = exp.runs[-1]

            assert run.status == "completed"
            best_points = run.metrics.get("best_fitness", [])
            mean_points = run.metrics.get("mean_fitness", [])
            # One point per generation, numbered from 1.
            assert [p.step for p in best_points] == [1, 2]
            assert len(mean_points) == 2
            # Tournament mode scores are ELO ratings anchored near 1500.
            assert all(p.value > 1000 for p in best_points)

            # Params were recorded so `crp report --type experiment` is useful.
            assert run.params.get("population_size") == 6
            assert run.params.get("max_generations") == 2

            # The artifact must point back at the real run directory.
            assert any(runs in a for a in run.artifacts)

    def test_tracking_is_off_by_default(self):
        """Opt-in: ordinary runs must not start creating tracker state."""
        config = TrainingConfig()
        assert config.enable_experiment_tracking is False


class TestReportFormatsFromRealArtifacts:
    """All three report formats read the trainer's actual on-disk schema."""

    def _make_run(self, tmpdir: str) -> str:
        run_dir = os.path.join(tmpdir, "run_x")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({
                "generation": 3, "max_generations": 10,
                "best_score": 1542.7, "best_score_kind": "elo",
                "tournament_mode": True, "population_size": 8,
            }, f)
        with open(os.path.join(run_dir, "fitness_history.json"), "w") as f:
            json.dump({"best": [0.51, 0.62, 0.73],
                       "mean": [0.40, 0.48, 0.55]}, f)
        return run_dir

    def test_html_labels_elo_scores_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._make_run(tmp)
            out = ReportGenerator(output_dir=tmp).generate_training_report(run_dir)
            html = open(out).read()
            assert "1542.7" in html or "1542.7000" in html
            assert "Best Score (ELO)" in html

    def test_markdown_and_json_carry_the_same_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._make_run(tmp)
            gen = ReportGenerator(output_dir=tmp)
            data = gen.load_run_data(run_dir)

            md_path = gen.generate_markdown_report(data["metrics"], "training", "r.md")
            json_path = gen.generate_json_report(
                {"type": "training", **data["metrics"]}, "r.json")

            md = open(md_path).read()
            # generate_json_report wraps the payload under "data".
            payload = json.load(open(json_path))["data"]

            assert payload["best_fitness"] == 1542.7
            assert payload["actual_generations"] == 3
            assert "1542" in md
            assert data["fitness_history"]["best"][-1] == 0.73
