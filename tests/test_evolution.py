"""Tests for evolutionary algorithm operators.

Tests:
- Selection strategies
- Crossover operators
- Mutation operators
- Population management
- Evolution strategy integration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models import (
    TournamentSelection,
    RankSelection,
    RouletteSelection,
    BlendCrossover,
    SinglePointCrossover,
    UniformCrossover,
    GaussianMutation,
    UniformMutation,
    AdaptiveMutation,
    EvolutionStrategy,
    EvolutionConfig,
    Population,
)


class TestTournamentSelection:
    """Tests for tournament selection."""

    def test_selects_fittest_in_tournament(self):
        """Test that tournament selection picks the fittest."""
        fitnesses = [1.0, 5.0, 3.0, 8.0, 2.0]
        population = list(range(len(fitnesses)))

        selector = TournamentSelection(tournament_size=3, rng=np.random.RandomState(42))
        p1, p2 = selector.select_parents(population, fitnesses)

        assert p1 in range(len(fitnesses))
        assert p2 in range(len(fitnesses))
        assert p1 != p2

    def test_selects_different_parents(self):
        """Test that two different parents are selected."""
        fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
        population = list(range(len(fitnesses)))

        selector = TournamentSelection(tournament_size=2, rng=np.random.RandomState(42))
        parents = set()
        for _ in range(100):
            p1, p2 = selector.select_parents(population, fitnesses)
            parents.add((min(p1, p2), max(p1, p2)))

        assert len(parents) > 1  # Should select different pairs


class TestRankSelection:
    """Tests for rank-based selection."""

    def test_selects_parents(self):
        """Test rank selection produces valid parents."""
        fitnesses = [1.0, 5.0, 3.0, 8.0, 2.0]
        population = list(range(len(fitnesses)))

        selector = RankSelection(weight_exponent=1.5, rng=np.random.RandomState(42))
        p1, p2 = selector.select_parents(population, fitnesses)

        assert p1 in range(len(fitnesses))
        assert p2 in range(len(fitnesses))


class TestRouletteSelection:
    """Tests for roulette wheel selection."""

    def test_selects_parents(self):
        """Test roulette selection produces valid parents."""
        fitnesses = [1.0, 5.0, 3.0, 8.0, 2.0]
        population = list(range(len(fitnesses)))

        selector = RouletteSelection(rng=np.random.RandomState(42))
        p1, p2 = selector.select_parents(population, fitnesses)

        assert p1 in range(len(fitnesses))
        assert p2 in range(len(fitnesses))


class TestBlendCrossover:
    """Tests for blend crossover."""

    def test_crossover_produces_offspring(self):
        """Test crossover produces valid offspring."""
        p1 = np.array([1.0, 2.0, 3.0, 4.0])
        p2 = np.array([5.0, 6.0, 7.0, 8.0])

        crossover = BlendCrossover(rng=np.random.RandomState(42))
        c1, c2 = crossover.crossover(p1, p2, alpha=0.5)

        assert c1.shape == p1.shape
        assert c2.shape == p2.shape
        assert not np.array_equal(c1, p1)
        assert not np.array_equal(c2, p2)

    def test_offspring_between_parents(self):
        """Test that offspring are between parent values (for alpha=0)."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([10.0, 10.0, 10.0])

        crossover = BlendCrossover(rng=np.random.RandomState(42))
        c1, c2 = crossover.crossover(p1, p2, alpha=0.0)

        # Offspring should be between parents (with small noise)
        assert np.all(c1 >= -0.1) and np.all(c1 <= 10.1)
        assert np.all(c2 >= -0.1) and np.all(c2 <= 10.1)


class TestSinglePointCrossover:
    """Tests for single-point crossover."""

    def test_crossover_produces_offspring(self):
        """Test single-point crossover."""
        p1 = np.array([1.0, 2.0, 3.0, 4.0])
        p2 = np.array([5.0, 6.0, 7.0, 8.0])

        crossover = SinglePointCrossover(rng=np.random.RandomState(42))
        c1, c2 = crossover.crossover(p1, p2)

        assert c1.shape == p1.shape
        assert c2.shape == p2.shape
        assert not np.array_equal(c1, p1)
        assert not np.array_equal(c2, p2)

    def test_crossover_point_effect(self):
        """Test that crossover actually swaps values."""
        p1 = np.array([1.0, 1.0, 1.0, 1.0])
        p2 = np.array([2.0, 2.0, 2.0, 2.0])

        crossover = SinglePointCrossover(rng=np.random.RandomState(42))
        c1, c2 = crossover.crossover(p1, p2)

        # c1 should have some 1s and some 2s
        assert 1.0 in c1
        assert 2.0 in c1


class TestGaussianMutation:
    """Tests for Gaussian mutation."""

    def test_mutates_weights(self):
        """Test Gaussian mutation modifies weights."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])

        mutation = GaussianMutation(rng=np.random.RandomState(42))
        mutated = mutation.mutate(weights, rate=1.0, std=0.1)

        # All weights should be mutated (rate=1.0)
        assert not np.array_equal(weights, mutated)
        assert mutated.shape == weights.shape

    def test_no_mutation_with_zero_rate(self):
        """Test that zero mutation rate preserves weights."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])

        mutation = GaussianMutation(rng=np.random.RandomState(42))
        mutated = mutation.mutate(weights, rate=0.0, std=0.1)

        assert np.array_equal(weights, mutated)


class TestUniformMutation:
    """Tests for uniform mutation."""

    def test_mutates_weights(self):
        """Test uniform mutation."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])

        mutation = UniformMutation(rng=np.random.RandomState(42))
        mutated = mutation.mutate(weights, rate=1.0, std=0.1)

        assert not np.array_equal(weights, mutated)


class TestAdaptiveMutation:
    """Tests for adaptive mutation."""

    def test_adjusts_mutation_strength(self):
        """Test that adaptive mutation adjusts std."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])

        mutation = AdaptiveMutation(
            min_std=0.01, max_std=0.5,
            rng=np.random.RandomState(42),
        )

        # Mutate with improving fitness (should decrease std)
        m1 = mutation.mutate(weights, rate=1.0, current_fitness=10.0)
        assert mutation.current_std < 0.101  # Should decrease from default

        # Mutate multiple times with no improvement to trigger increase
        for _ in range(6):
            mutation.mutate(weights, rate=1.0, current_fitness=5.0)
        # After 6+ no-improvement calls, std should start increasing
        assert mutation.current_std > 0.095  # Should be increasing


class TestEvolutionStrategy:
    """Tests for the evolution strategy."""

    def test_evolve_produces_same_size(self):
        """Test evolution produces population of same size."""
        config = EvolutionConfig(population_size=10)
        strategy = EvolutionStrategy(config)

        population = [np.random.randn(100) for _ in range(10)]
        fitnesses = [float(i) for i in range(10)]

        offspring, _ = strategy.evolve(population, fitnesses)
        assert len(offspring) == 10

    def test_elitism_preserves_best(self):
        """Test that elitism preserves the best agent."""
        config = EvolutionConfig(
            population_size=10,
            elite_count=2,
            elite_preservation=True,
        )
        strategy = EvolutionStrategy(config)

        population = [np.random.randn(100) for _ in range(10)]
        fitnesses = [float(i) for i in range(10)]

        best_idx = fitnesses.index(max(fitnesses))
        best_weights = population[best_idx].copy()

        offspring, _ = strategy.evolve(population, fitnesses)

        # Best agent should be preserved
        found = False
        for w in offspring:
            if np.allclose(w, best_weights):
                found = True
                break
        assert found

    def test_no_elitism(self):
        """Test evolution without elitism."""
        config = EvolutionConfig(
            population_size=10,
            elite_preservation=False,
        )
        strategy = EvolutionStrategy(config)

        population = [np.random.randn(100) for _ in range(10)]
        fitnesses = [float(i) for i in range(10)]

        offspring, _ = strategy.evolve(population, fitnesses)
        assert len(offspring) == 10


class TestPopulation:
    """Tests for population management."""

    def test_initialize_population(self):
        """Test population initialization."""
        pop = Population(population_size=10, elite_count=2)
        pop.initialize(seed=42)

        assert len(pop) == 10
        assert pop.generation == 0

    def test_evaluate_and_rank(self):
        """Test fitness evaluation and ranking."""
        pop = Population(population_size=5, elite_count=2)
        pop.initialize(seed=42)

        fitnesses = [3.0, 1.0, 4.0, 2.0, 5.0]
        pop.evaluate(fitnesses)

        best = pop.get_best()
        assert best is not None
        assert best.fitness == 5.0
        assert best.rank == 1

    def test_get_elite(self):
        """Test elite retrieval."""
        pop = Population(population_size=10, elite_count=3)
        pop.initialize(seed=42)

        fitnesses = [float(i) for i in range(10)]
        pop.evaluate(fitnesses)

        elites = pop.get_elite()
        assert len(elites) == 3
        assert elites[0].fitness > elites[1].fitness > elites[2].fitness

    def test_get_statistics(self):
        """Test population statistics."""
        pop = Population(population_size=5, elite_count=2)
        pop.initialize(seed=42)

        fitnesses = [1.0, 3.0, 2.0, 5.0, 4.0]
        pop.evaluate(fitnesses)

        stats = pop.get_statistics()
        assert stats["best_fitness"] == 5.0
        assert stats["min_fitness"] == 1.0
        assert stats["population_size"] == 5

    def test_get_diversity(self):
        """Test diversity computation."""
        pop = Population(population_size=5, elite_count=2)
        pop.initialize(seed=42)

        fitnesses = [float(i) for i in range(5)]
        pop.evaluate(fitnesses)

        diversity = pop.get_diversity()
        assert diversity >= 0

    def test_get_population_weights(self):
        """Test getting all population weights."""
        pop = Population(population_size=3, elite_count=1)
        pop.initialize(seed=42)

        weights = pop.get_population_weights()
        assert len(weights) == 3
        assert all(w.shape[0] > 0 for w in weights)

    def test_set_population_weights(self):
        """Test setting population weights."""
        pop = Population(population_size=3, elite_count=1)
        pop.initialize(seed=42)

        # Get actual weight size from the first agent
        actual_size = len(pop.agents[0].agent.get_weights())
        new_weights = [np.random.randn(actual_size) for _ in range(3)]
        pop.set_population_weights(new_weights)

        current_weights = pop.get_population_weights()
        for i in range(3):
            assert np.allclose(current_weights[i], new_weights[i])

    def test_fitness_history(self):
        """Test fitness history tracking."""
        pop = Population(population_size=5, elite_count=2)
        pop.initialize(seed=42)

        for gen in range(3):
            fitnesses = [float(i) + gen for i in range(5)]
            pop.evaluate(fitnesses)

        assert len(pop.fitness_history["best"]) == 3


class TestNewFeatures:
    """Tests for new evolution features."""

    def test_tournament_elite_selection(self):
        """Test tournament elite selection favors elites."""
        from src.models import TournamentEliteSelection
        fitnesses = [1.0, 10.0, 1.0, 1.0, 1.0]
        population = list(range(len(fitnesses)))
        selector = TournamentEliteSelection(tournament_size=3, rng=np.random.RandomState(42))
        p1, p2 = selector.select_parents(population, fitnesses)
        assert p1 in range(len(fitnesses))
        assert p2 in range(len(fitnesses))

    def test_arithmetic_crossover(self):
        """Test arithmetic crossover."""
        from src.models import ArithmeticCrossover
        p1 = np.array([1.0, 2.0, 3.0, 4.0])
        p2 = np.array([5.0, 6.0, 7.0, 8.0])
        crossover = ArithmeticCrossover(rng=np.random.RandomState(42))
        c1, c2 = crossover.crossover(p1, p2, alpha=0.5)
        assert c1.shape == p1.shape
        assert c2.shape == p2.shape
        # With alpha=0.5, children should be close to average
        expected = (p1 + p2) / 2
        assert np.allclose(c1, expected, atol=0.1)

    def test_diversity_tracker(self):
        """Test diversity tracking."""
        from src.models import DiversityTracker
        tracker = DiversityTracker(sample_size=5)
        weights = [np.random.randn(50) for _ in range(5)]
        diversity = tracker.compute_diversity(weights, rng=np.random.RandomState(42))
        assert diversity >= 0
        tracker.record_diversity(diversity)
        stats = tracker.get_diversity_stats()
        assert "mean" in stats
        assert "std" in stats

    def test_speciation(self):
        """Test speciation groups similar agents."""
        from src.models import DiversityTracker
        tracker = DiversityTracker()
        # Two distinct groups
        group1 = [np.array([1.0] * 50) + np.random.randn(50) * 0.01 for _ in range(3)]
        group2 = [np.array([-1.0] * 50) + np.random.randn(50) * 0.01 for _ in range(3)]
        all_weights = group1 + group2
        species = tracker.speciate(all_weights, threshold=0.5)
        total_members = sum(len(m) for m in species.values())
        assert total_members == len(all_weights)

    def test_evolution_with_diversity(self):
        """Test evolution with diversity tracking enabled."""
        config = EvolutionConfig(
            population_size=10,
            elite_count=2,
            diversity_preservation=True,
            diversity_threshold=0.5,
        )
        strategy = EvolutionStrategy(config)
        population = [np.random.randn(100) for _ in range(10)]
        fitnesses = [float(i) for i in range(10)]
        offspring, info = strategy.evolve(population, fitnesses)
        assert len(offspring) == 10

    def test_population_speciation(self):
        """Test population speciation."""
        pop = Population(population_size=10, elite_count=2)
        pop.initialize(seed=42)
        fitnesses = [float(i) for i in range(10)]
        pop.evaluate(fitnesses)
        species = pop.speciate(threshold=0.5)
        assert isinstance(species, dict)
        total = sum(len(m) for m in species.values())
        assert total == 10

    def test_population_diversity_update(self):
        """Test population diversity update."""
        pop = Population(population_size=10, elite_count=2)
        pop.initialize(seed=42)
        fitnesses = [float(i) for i in range(10)]
        pop.evaluate(fitnesses)
        diversity = pop.update_diversity()
        assert diversity >= 0
        assert len(pop.diversity_history) > 0

    def test_population_species_diversity(self):
        """Test per-species diversity."""
        pop = Population(population_size=10, elite_count=2)
        pop.initialize(seed=42)
        fitnesses = [float(i) for i in range(10)]
        pop.evaluate(fitnesses)
        pop.speciate(threshold=0.5)
        species_div = pop.get_species_diversity()
        assert isinstance(species_div, dict)

    def test_evolution_config_summary(self):
        """Test evolution config summary."""
        config = EvolutionConfig()
        strategy = EvolutionStrategy(config)
        summary = strategy.get_config_summary()
        assert "selection" in summary
        assert "crossover" in summary
        assert "mutation" in summary
        assert summary["selection"] == "TOURNAMENT"
        assert summary["crossover"] == "BLEND"
        assert summary["mutation"] == "GAUSSIAN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

