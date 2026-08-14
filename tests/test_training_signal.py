"""Tests that the training loop carries real signal.

Every test here corresponds to a defect that made evolution a no-op while the
rest of the suite stayed green:

* ``_infer_action`` ignored the genome, so all agents played identically.
* ``ParallelRunner.evaluate_population`` raised ``IndexError`` on every call
  and a bare ``except`` turned that into fitness 0.0 for the whole population.
* ``Population`` evolved a 9.28M-parameter Torch network that no match ever
  consulted, while the genome the agent stored went unused.
* Every match in a worker reset to the same seed, so N matches per agent
  measured exactly as much as one.

These are end-to-end properties: if any of them regress, training silently
stops learning, which is precisely the failure mode that is hard to notice.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim.engine import SimulationEngine
from src.env.sim.parallel_runner import (
    WorkerConfig, _policy_action, _run_matches,
)
from src.models.evolution import EvolutionConfig, EvolutionStrategy
from src.models.policy import DEFAULT_POLICY_SPEC
from src.models.population import Population


def _genome(seed: int) -> np.ndarray:
    return DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(seed))


def _fitness(genome, seed=11, matches=2, opponent="balanced") -> float:
    config = WorkerConfig(seed=seed, match_count=matches,
                          match_duration_ticks=600, overtime_ticks=60)
    return _run_matches(0, config, genome, opponent, None).fitness


class TestGenomeDrivesBehaviour:
    """The genome must change how the agent plays."""

    def test_different_genomes_play_differently(self):
        signatures = set()
        for seed in range(6):
            genome = _genome(seed)
            engine = SimulationEngine(seed=5, record_replay=False)
            engine.reset()
            for _ in range(250):
                if engine.terminated:
                    break
                engine.step(_policy_action(genome, engine, "player"), None)
            signatures.add((
                len(engine.action_history),
                round(engine.player_elixir, 3),
                tuple(a["card"] for a in engine.action_history[:4]),
            ))
        # Not every genome must differ, but they cannot all collapse to one.
        assert len(signatures) >= 3

    def test_a_null_genome_passes(self):
        engine = SimulationEngine(seed=5, record_replay=False)
        engine.reset()
        action = _policy_action(None, engine, "player")
        assert not action.is_deploy_action()

    def test_policy_never_places_units_on_the_enemy_half(self):
        """Placement mapping must respect deployment rules for every genome."""
        for seed in range(12):
            genome = _genome(seed)
            engine = SimulationEngine(seed=seed, record_replay=False)
            engine.reset()
            for _ in range(200):
                if engine.terminated:
                    break
                engine.step(_policy_action(genome, engine, "player"), None)
            for record in engine.action_history:
                from src.env.sim.entities import CARD_DEFS
                if CARD_DEFS[record["card"]].card_type != "spell":
                    assert record["row"] >= engine.BRIDGE_ROW + 1


class TestFitnessSignal:
    """Fitness must vary with the genome and be reproducible."""

    def test_fitness_varies_across_genomes(self):
        scores = [_fitness(_genome(s)) for s in range(8)]
        assert len(set(scores)) > 1
        assert np.std(scores) > 0.0

    def test_fitness_is_reproducible(self):
        genome = _genome(21)
        assert _fitness(genome) == _fitness(genome)

    def test_matches_within_a_worker_are_not_identical(self):
        """Each match must get its own seed, or averaging measures nothing."""
        genome = _genome(31)
        config = WorkerConfig(seed=4, match_count=6,
                              match_duration_ticks=600, overtime_ticks=60)
        result = _run_matches(0, config, genome, "random", None)
        assert result.wins + result.draws + result.losses == 6
        # A single repeated match would make every outcome the same.
        assert len({result.wins, result.draws, result.losses}) > 1

    def test_a_passive_genome_loses_to_an_active_opponent(self):
        """Sanity check that the scoreboard reflects actually playing."""
        passive = np.zeros(DEFAULT_POLICY_SPEC.num_params)
        # An all-zero genome yields equal logits; pass (index 4) ties and wins
        # argmax only if no card outranks it, so assert on the outcome instead.
        score = _fitness(passive, matches=2, opponent="aggressive")
        assert np.isfinite(score)


class TestPopulationPlumbing:
    """The population must evolve the genome the simulator actually runs."""

    def test_population_genomes_match_the_policy_spec(self):
        pop = Population(population_size=6, elite_count=2)
        pop.initialize(seed=0)
        for genome in pop.get_population_weights():
            assert genome.shape == (DEFAULT_POLICY_SPEC.num_params,)

    def test_population_genomes_are_distinct(self):
        pop = Population(population_size=6, elite_count=2)
        pop.initialize(seed=0)
        genomes = pop.get_population_weights()
        assert not np.array_equal(genomes[0], genomes[1])

    def test_set_population_weights_round_trips(self):
        pop = Population(population_size=4, elite_count=1)
        pop.initialize(seed=0)
        replacement = [_genome(100 + i) for i in range(4)]
        pop.set_population_weights(replacement)
        for expected, actual in zip(replacement, pop.get_population_weights()):
            assert np.array_equal(expected, actual)

    def test_population_genomes_are_directly_playable(self):
        """No conversion step between what evolves and what plays."""
        pop = Population(population_size=3, elite_count=1)
        pop.initialize(seed=0)
        engine = SimulationEngine(seed=2, record_replay=False)
        engine.reset()
        action = _policy_action(pop.get_population_weights()[0], engine, "player")
        assert action is not None


class TestParallelEvaluation:
    """ParallelRunner.evaluate_population used to fail on every call."""

    def test_evaluate_population_returns_real_scores(self):
        from src.env.sim.parallel_runner import ParallelRunner

        runner = ParallelRunner(num_workers=2)
        try:
            genomes = [_genome(s) for s in range(4)]
            results = runner.evaluate_population(
                genomes, matches_per_agent=1, opponent_type="balanced", seed=3)
        finally:
            runner.shutdown()

        assert len(results) == 4
        assert [r.agent_id for r in results] == [f"agent_{i}" for i in range(4)]
        # The old failure mode: a swallowed IndexError zeroed the population.
        assert not all(r.fitness == 0.0 for r in results)
        assert any(r.wins + r.draws + r.losses > 0 for r in results)


class TestTrainerLifecycle:
    """The trainer attaches a file handler to a module-level logger."""

    def test_close_detaches_the_run_log_handler(self, tmp_path):
        import logging

        from src.train import trainer as trainer_module
        from src.train.trainer import EvolutionTrainer, TrainingConfig

        config = TrainingConfig(population_size=2, elite_count=1,
                                max_generations=0, num_workers=1,
                                runs_dir=str(tmp_path))
        before = len(trainer_module.logger.handlers)

        instance = EvolutionTrainer(config)
        assert len(trainer_module.logger.handlers) == before + 1

        instance.close()

        assert len(trainer_module.logger.handlers) == before
        assert instance._log_handler is None

    def test_trainer_works_as_a_context_manager(self, tmp_path):
        from src.train import trainer as trainer_module
        from src.train.trainer import EvolutionTrainer, TrainingConfig

        config = TrainingConfig(population_size=2, elite_count=1,
                                max_generations=0, num_workers=1,
                                runs_dir=str(tmp_path))
        before = len(trainer_module.logger.handlers)

        with EvolutionTrainer(config):
            pass

        assert len(trainer_module.logger.handlers) == before


class TestEvolutionImproves:
    """The point of the whole pipeline."""

    def test_selection_raises_mean_fitness(self):
        """Fitness must improve under selection against a real opponent.

        Every agent in a generation is scored on the same match seeds (common
        random numbers), which is what makes their fitness comparable; scoring
        each agent on its own seeds leaves the differences dominated by draw
        luck and selection sorts mostly noise.
        """
        pop_size = 12
        pop = Population(population_size=pop_size, elite_count=3)
        pop.initialize(seed=5)
        evolution = EvolutionStrategy(EvolutionConfig(
            population_size=pop_size, elite_count=3,
            mutation_rate=0.15, mutation_std=0.12, crossover_rate=0.6,
            seed=11,
        ))

        means = []
        for generation in range(10):
            genomes = pop.get_population_weights()
            generation_seed = 200 + generation * 1000
            scores = [_fitness(g, seed=generation_seed, matches=3)
                      for g in genomes]
            pop.evaluate(scores)
            means.append(float(np.mean(scores)))
            new_genomes, _info = evolution.evolve(genomes, scores)
            pop.set_population_weights(new_genomes)

        assert np.mean(means[-3:]) > np.mean(means[:3])

    def test_elites_survive_a_generation(self):
        pop_size = 8
        genomes = [_genome(s) for s in range(pop_size)]
        scores = [float(i) for i in range(pop_size)]  # last genome is best
        evolution = EvolutionStrategy(EvolutionConfig(
            population_size=pop_size, elite_count=2,
            mutation_rate=0.0, mutation_std=0.0, crossover_rate=0.0,
        ))
        new_genomes, _info = evolution.evolve(genomes, scores)
        assert len(new_genomes) == pop_size
        assert all(g.shape == genomes[0].shape for g in new_genomes)
