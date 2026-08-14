"""Regression tests for genetic-algorithm operators.

Each class covers a defect that let training run to completion while quietly
degrading -- or inverting -- the search.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.evolution import (
    AdaptiveMutation, EvolutionConfig, EvolutionStrategy, GaussianMutation,
    RankSelection, RouletteSelection, TournamentEliteSelection,
    TournamentSelection, UniformMutation,
)
from src.models.policy import DEFAULT_POLICY_SPEC

SELECTORS = [
    ("tournament", lambda: TournamentSelection(tournament_size=3)),
    ("rank", lambda: RankSelection()),
    ("roulette", lambda: RouletteSelection()),
    ("tournament_elite", lambda: TournamentEliteSelection(tournament_size=3)),
]


class TestSelectionFavoursFitness:
    """Every selector must prefer fitter individuals."""

    @pytest.mark.parametrize("name,factory", SELECTORS)
    def test_fittest_is_selected_more_than_weakest(self, name, factory):
        """Rank selection previously drew by array position, not by fitness.

        With `probs` indexed by rank but the raw draw returned as a population
        index, a worst-first population had its *worst* members bred most
        often -- selection pressure pointed backwards.
        """
        fitnesses = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0]
        population = [np.zeros(3) for _ in fitnesses]
        rng = np.random.RandomState(0)
        selector = factory()

        counts = np.zeros(len(fitnesses), dtype=int)
        for _ in range(2000):
            first, _second = selector.select_parents(population, fitnesses, rng)
            counts[first] += 1

        best, worst = len(fitnesses) - 1, 0
        assert counts[best] > counts[worst], (
            f"{name} selected the weakest individual more often than the fittest"
        )

    @pytest.mark.parametrize("name,factory", SELECTORS)
    def test_selection_is_order_independent(self, name, factory):
        """Shuffling the population must not change who is favoured."""
        fitnesses = [100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]  # best first
        population = [np.zeros(3) for _ in fitnesses]
        rng = np.random.RandomState(1)
        selector = factory()

        counts = np.zeros(len(fitnesses), dtype=int)
        for _ in range(2000):
            first, _second = selector.select_parents(population, fitnesses, rng)
            counts[first] += 1

        assert counts[0] > counts[7], (
            f"{name} ignores fitness when the fittest sits at index 0"
        )


class TestSelectionTerminates:
    """Parent re-draws must be bounded."""

    def test_tournament_size_at_population_size_does_not_hang(self):
        """A full-population tournament is deterministic, so the old
        `while second == first` retry could never terminate."""
        fitnesses = [1.0, 2.0, 3.0, 4.0]
        population = [np.zeros(2) for _ in fitnesses]
        selector = TournamentSelection(tournament_size=len(fitnesses))

        first, second = selector.select_parents(
            population, fitnesses, np.random.RandomState(0))

        assert first == second == 3  # fittest, unavoidably duplicated

    def test_single_individual_population(self):
        selector = TournamentSelection(tournament_size=3)
        first, second = selector.select_parents(
            [np.zeros(2)], [1.0], np.random.RandomState(0))
        assert first == second == 0

    def test_degenerate_roulette_distribution_terminates(self):
        fitnesses = [0.0, 0.0, 0.0, 1000.0]
        population = [np.zeros(2) for _ in fitnesses]
        selector = RouletteSelection()
        first, second = selector.select_parents(
            population, fitnesses, np.random.RandomState(0))
        assert first in range(4) and second in range(4)


class TestMutationOperatorInterface:
    """All mutation operators share one call signature."""

    @pytest.mark.parametrize("factory", [GaussianMutation, UniformMutation,
                                         AdaptiveMutation])
    def test_accepts_rng_in_the_same_position(self, factory):
        """AdaptiveMutation omitted `rng`, so a positional call bound the
        RandomState to `current_fitness` and failed on a float comparison."""
        operator = factory()
        weights = np.ones(16)
        result = operator.mutate(weights, 0.5, 0.1, np.random.RandomState(0))
        assert result.shape == weights.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("factory", [GaussianMutation, UniformMutation,
                                         AdaptiveMutation])
    def test_rng_makes_mutation_reproducible(self, factory):
        operator = factory()
        weights = np.ones(64)
        first = operator.mutate(weights, 0.5, 0.1, np.random.RandomState(7))
        second = operator.mutate(weights, 0.5, 0.1, np.random.RandomState(7))
        assert np.array_equal(first, second)

    def test_mutation_leaves_the_parent_untouched(self):
        weights = np.ones(32)
        original = weights.copy()
        GaussianMutation().mutate(weights, 1.0, 0.5, np.random.RandomState(0))
        assert np.array_equal(weights, original)


class TestEvolutionReproducibility:
    """A seeded config must reproduce a run exactly."""

    def _evolve(self, seed):
        genomes = [DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(i))
                   for i in range(8)]
        fitnesses = [float(i) for i in range(8)]
        strategy = EvolutionStrategy(EvolutionConfig(
            population_size=8, elite_count=2, seed=seed))
        offspring, _info = strategy.evolve(genomes, fitnesses)
        return np.concatenate(offspring)

    def test_same_seed_reproduces_the_generation(self):
        """EvolutionStrategy used an entropy-seeded RandomState, so
        TrainingConfig.seed had no effect on evolution at all."""
        assert np.array_equal(self._evolve(7), self._evolve(7))

    def test_different_seeds_diverge(self):
        assert not np.array_equal(self._evolve(7), self._evolve(8))

    def test_unseeded_config_still_works(self):
        assert self._evolve(None).size > 0

    def test_offspring_keep_the_genome_shape(self):
        genomes = [DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(i))
                   for i in range(6)]
        strategy = EvolutionStrategy(EvolutionConfig(
            population_size=6, elite_count=2, seed=3))
        offspring, _info = strategy.evolve(genomes, [1.0] * 6)
        assert len(offspring) == 6
        assert all(child.shape == genomes[0].shape for child in offspring)

    def test_trainer_seed_reaches_the_evolution_strategy(self):
        from src.train.trainer import TrainingConfig, EvolutionTrainer
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(population_size=4, elite_count=1,
                                    max_generations=0, num_workers=1,
                                    runs_dir=tmp, seed=1234)
            with EvolutionTrainer(config) as trainer:
                assert trainer.evolution.config.seed == 1234
