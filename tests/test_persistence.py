"""Tests that a trained agent survives being written to disk and read back.

The evolved policy genome is the trained artefact. Saving the Torch network
instead (which the evolutionary path never touches, and which is freshly
initialised) silently discards the result of a training run -- the checkpoint
loads, reports no error, and plays like a random agent.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.env.sim.engine import SimulationEngine
from src.env.sim.parallel_runner import _policy_action
from src.models.agent import AgentConfig, EvolutionaryAgent
from src.models.policy import DEFAULT_POLICY_SPEC
from src.train.trainer import EvolutionTrainer, TrainingConfig


def _genome(seed: int) -> np.ndarray:
    return DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(seed))


def _played_actions(genome, seed=5, ticks=150):
    engine = SimulationEngine(seed=seed, record_replay=False)
    engine.reset()
    for _ in range(ticks):
        if engine.terminated:
            break
        engine.step(_policy_action(genome, engine, "player"), None)
    return [(a["tick"], a["card"], round(a["col"], 4), round(a["row"], 4))
            for a in engine.action_history]


class TestAgentCheckpointRoundTrip:

    def test_genome_survives_save_and_load(self):
        genome = _genome(3)
        agent = EvolutionaryAgent(weights=genome)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            agent.save_checkpoint(path)
            restored = EvolutionaryAgent()
            restored.load_checkpoint(path)

        assert np.array_equal(np.asarray(restored.weights), genome)

    def test_restored_agent_plays_identically(self):
        genome = _genome(4)
        agent = EvolutionaryAgent(weights=genome)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            agent.save_checkpoint(path)
            restored = EvolutionaryAgent()
            restored.load_checkpoint(path)

        assert _played_actions(restored.weights) == _played_actions(genome)
        assert _played_actions(genome) != []  # the probe must be meaningful

    def test_loading_a_genome_does_not_build_a_torch_network(self):
        """Restoring a policy should not pay for, or fabricate, a CNN+LSTM."""
        agent = EvolutionaryAgent(weights=_genome(5))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            agent.save_checkpoint(path)
            restored = EvolutionaryAgent()
            restored.load_checkpoint(path)
            assert not restored.has_network

    def test_checkpoint_records_which_parameters_it_holds(self):
        agent = EvolutionaryAgent(weights=_genome(6))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            agent.save_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)

        assert payload["param_kind"] == "genome"
        assert np.asarray(payload["genome"]).size == DEFAULT_POLICY_SPEC.num_params
        # Legacy readers key off "weights"; it must not be the untrained net.
        assert np.asarray(payload["weights"]).size == DEFAULT_POLICY_SPEC.num_params

    def test_network_only_agent_still_round_trips(self):
        """Agents without a genome keep the original Torch behaviour."""
        agent = EvolutionaryAgent(config=AgentConfig(architecture="cnn_lstm"))
        original = agent.get_weights()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            agent.save_checkpoint(path)
            restored = EvolutionaryAgent(config=AgentConfig(architecture="cnn_lstm"))
            restored.load_checkpoint(path)

        assert np.allclose(restored.get_weights(), original)


class TestTrainerPersistsTheBestAgent:

    def _train(self, tmpdir, generations=3):
        config = TrainingConfig(
            population_size=8, elite_count=2, max_generations=generations,
            matches_per_agent=2, num_workers=2, match_duration="short",
            opponent_type="balanced", runs_dir=tmpdir, seed=1,
            checkpoint_interval=100, curriculum_learning=False,
            diversity_preservation=False,
        )
        trainer = EvolutionTrainer(config)
        with trainer:
            trainer.train()
            return (np.array(trainer.best_genome, copy=True),
                    trainer.best_fitness)

    def test_saved_agent_is_the_trained_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            best_genome, _fitness = self._train(tmp)
            path = os.path.join(tmp, "best", "best_agent.pt")
            assert os.path.exists(path)

            restored = EvolutionaryAgent()
            restored.load_checkpoint(path)

            assert np.array_equal(np.asarray(restored.weights), best_genome)

    def test_best_genome_is_snapshotted_not_referenced(self):
        """AgentRecords are reused in place by set_population_weights, so a
        live reference to the best record stops describing the best agent as
        soon as the next generation is written into it."""
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(
                population_size=8, elite_count=2, max_generations=4,
                matches_per_agent=2, num_workers=2, match_duration="short",
                opponent_type="balanced", runs_dir=tmp, seed=1,
                checkpoint_interval=100, curriculum_learning=False,
                diversity_preservation=False,
            )
            with EvolutionTrainer(config) as trainer:
                trainer.train()
                snapshot = np.asarray(trainer.best_genome)
                path = os.path.join(tmp, "best", "best_agent.pt")

            restored = EvolutionaryAgent()
            restored.load_checkpoint(path)
            # The file must agree with the snapshot, whatever later
            # generations did to the population slot it came from.
            assert np.array_equal(np.asarray(restored.weights), snapshot)

    def test_evaluate_script_can_load_the_result(self):
        from scripts.evaluate import load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            best_genome, _fitness = self._train(tmp)
            path = os.path.join(tmp, "best", "best_agent.pt")
            assert np.array_equal(load_checkpoint(path), best_genome)

    def test_evaluate_script_rejects_network_weights(self):
        """A Torch checkpoint cannot be played by the simulator; say so
        rather than handing back an unusable vector."""
        from scripts.evaluate import load_checkpoint

        agent = EvolutionaryAgent(config=AgentConfig(architecture="cnn_lstm"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "net.pt")
            agent.save_checkpoint(path)
            with pytest.raises(ValueError, match="policy genomes"):
                load_checkpoint(path)
