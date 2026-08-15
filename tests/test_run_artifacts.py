"""Tests for what a training run leaves behind.

Every case here comes from inspecting two real runs on disk rather than from
reading code. They all passed the existing suite while producing runs that were
awkward or impossible to work with:

* ``best_agent.pt`` was **108 MB** for an 18 KB genome, because tracking a new
  best built the lazily-created 9.28M-parameter Torch network just to copy it.
* An 11-generation run had **no checkpoint at all** (the interval was 50), so
  19 minutes of training could not be continued.
* ``training.log`` was **zero bytes** in both runs -- the logger level was never
  set, so every INFO record was dropped.
* ``metrics.json`` claimed ``population_size: 240`` for a run training 24
  agents, because resuming kept the requested size instead of the real one.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.agent import AgentConfig, EvolutionaryAgent
from src.models.policy import DEFAULT_POLICY_SPEC
from src.serialization import load_checkpoint
from src.train.trainer import EvolutionTrainer, TrainingConfig

# A genome checkpoint should be tens of KB. The Torch network alone is 37MB.
MAX_AGENT_CHECKPOINT_BYTES = 2 * 1024 * 1024


def _config(run_dir, **overrides):
    params = dict(
        population_size=6, elite_count=2, max_generations=3, num_workers=2,
        match_duration="short", runs_dir=run_dir, seed=3,
        checkpoint_interval=50,          # deliberately longer than the run
        curriculum_learning=False, diversity_preservation=False,
        tournament_matches=2, hall_of_fame_size=2,
    )
    params.update(overrides)
    return TrainingConfig(**params)


@pytest.fixture(scope="module")
def finished_run():
    tmp = tempfile.mkdtemp()
    run = os.path.join(tmp, "run")
    os.makedirs(run)
    snapshots = []
    with EvolutionTrainer(_config(run)) as trainer:
        trainer.on_generation = snapshots.append
        trainer.train()
    return {"run": run, "snapshots": snapshots}


class TestAgentCheckpointSize:

    def test_best_agent_is_small(self, finished_run):
        path = os.path.join(finished_run["run"], "best", "best_agent.pt")
        size = os.path.getsize(path)
        assert size < MAX_AGENT_CHECKPOINT_BYTES, (
            f"best_agent.pt is {size / 1e6:.0f}MB; a policy genome is ~18KB, "
            f"so Torch network parameters are being written alongside it"
        )

    def test_no_network_weights_beside_a_genome(self, finished_run):
        payload = load_checkpoint(
            os.path.join(finished_run["run"], "best", "best_agent.pt"))
        assert payload["param_kind"] == "genome"
        assert payload["network_weights"] is None
        assert np.asarray(payload["genome"]).size == DEFAULT_POLICY_SPEC.num_params

    def test_update_best_snapshots_the_genome(self):
        """It used to call get_weights(), building the Torch network."""
        genome = DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(0))
        agent = EvolutionaryAgent(weights=genome)

        agent.update_best(1.0)

        assert np.array_equal(np.asarray(agent.best_weights), genome)
        assert not agent.has_network

    def test_network_agents_still_snapshot_their_weights(self):
        agent = EvolutionaryAgent(config=AgentConfig(architecture="cnn_mlp"))
        agent.forward(np.zeros((12, 64, 64), dtype=np.float32))  # build it
        agent.update_best(1.0)
        assert agent.best_weights is not None
        assert np.asarray(agent.best_weights).size > DEFAULT_POLICY_SPEC.num_params


class TestRunIsResumable:

    def test_a_checkpoint_exists_even_below_the_interval(self, finished_run):
        """A 3-generation run with interval 50 still has to leave something."""
        checkpoints = sorted(d for d in os.listdir(finished_run["run"])
                             if d.startswith("gen_"))
        assert checkpoints, "run ended with no checkpoint; it cannot be continued"

    def test_the_final_checkpoint_can_be_found(self, finished_run):
        found = EvolutionTrainer.find_population_checkpoint(finished_run["run"])
        assert found is not None and found.is_file()

    def test_a_stopped_run_is_still_resumable(self):
        """Stopping mid-run must not throw the work away."""
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "stopped")
            os.makedirs(run)
            trainer = EvolutionTrainer(_config(run, max_generations=20))

            def stop_after_two(snapshot):
                if snapshot["generation"] >= 2:
                    trainer.stop()

            with trainer:
                trainer.on_generation = stop_after_two
                trainer.train()

            assert EvolutionTrainer.find_population_checkpoint(run) is not None

    def test_checkpoints_are_not_duplicated(self):
        """The end-of-run save must not repeat one just written."""
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, "run")
            os.makedirs(run)
            with EvolutionTrainer(_config(run, max_generations=2,
                                          checkpoint_interval=1)) as trainer:
                trainer.train()
            checkpoints = sorted(d for d in os.listdir(run)
                                 if d.startswith("gen_"))
            assert checkpoints == ["gen_0001", "gen_0002"]


class TestRunRecord:

    def test_training_log_has_content(self, finished_run):
        path = os.path.join(finished_run["run"], "training.log")
        assert os.path.exists(path)
        with open(path, encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
        assert lines, "training.log is empty; the run left no record of itself"
        assert any("Generation" in line for line in lines)

    def test_closing_restores_the_logger_level(self):
        """A trainer must not permanently reconfigure logging for the process."""
        import logging

        from src.train import trainer as trainer_module

        before = trainer_module.logger.level
        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(_config(tmp, max_generations=1)) as trainer:
                assert trainer_module.logger.level <= logging.INFO
        assert trainer_module.logger.level == before

    def test_metrics_describe_the_run(self, finished_run):
        with open(os.path.join(finished_run["run"], "metrics.json")) as handle:
            metrics = json.load(handle)
        assert metrics["population_size"] == 6
        assert metrics["best_score_kind"] == "elo"
        assert metrics["generation"] == 3

    def test_resume_reports_the_real_population_size(self, finished_run):
        """Asking for a different size cannot change what the checkpoint holds,
        so the config must be reconciled rather than left misreporting."""
        with EvolutionTrainer(_config(finished_run["run"],
                                      population_size=240,
                                      resume_from=finished_run["run"],
                                      additional_generations=1)) as trainer:
            trainer._resume_from_checkpoint()
            assert trainer.config.population_size == 6
            assert len(trainer.population) == 6


class TestProgressSignal:
    """Tournament fitness cannot show progress; ratings can."""

    def test_snapshots_carry_ratings(self, finished_run):
        snapshots = finished_run["snapshots"]
        assert snapshots, "no progress was reported"
        for snapshot in snapshots:
            assert snapshot["champion_elo"] is not None
            assert snapshot["population_elo"] is not None

    def test_hall_of_fame_rating_appears_once_it_has_competed(self, finished_run):
        """Generation 1's champion is admitted *after* that tournament, so the
        hall of fame has no rating until it has actually played."""
        snapshots = finished_run["snapshots"]
        assert snapshots[0]["hall_of_fame_elo"] is None
        assert all(s["hall_of_fame_elo"] is not None for s in snapshots[1:])

    def test_population_rating_is_a_real_average(self, finished_run):
        snapshot = finished_run["snapshots"][-1]
        # Ratings start at 1500 and move from there.
        assert 1000 < snapshot["population_elo"] < 2000
        assert 1000 < snapshot["hall_of_fame_elo"] < 2000

    def test_progress_is_reported_every_generation(self, finished_run):
        generations = [s["generation"] for s in finished_run["snapshots"]]
        assert generations == [1, 2, 3]

    def test_rating_helpers_handle_an_empty_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(_config(tmp, max_generations=1)) as trainer:
                assert trainer._population_elo() is None
                assert trainer._hall_of_fame_elo() is None
