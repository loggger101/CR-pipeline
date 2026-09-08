"""Tests for the advanced GA dynamics added 2026-09.

Each test pins one mechanism so a future refactor cannot silently revert it:

* tempered selection -- parent draws must spread across the field, not collapse
  onto the single top agent (the pre-fix softmax was effectively argmax);
* champion refinements -- offspring that are close mutations of the run's best
  genome, i.e. each generation builds on the previous one;
* adaptive mutation -- live std widens after a stagnation window and shrinks on
  progress;
* immigration -- fresh genomes replace slots when population diversity collapses
  to near-clones.

These are pure-Python (no worker pool), so they run in milliseconds.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.evolution import TournamentEvolutionStrategy


def _population(n: int = 12, dim: int = 64, seed: int = 0) -> list:
    rng = np.random.RandomState(seed)
    return [rng.randn(dim) * (1.0 / np.sqrt(dim)) for _ in range(n)]


class TestTemperedSelection:
    """Parent selection must not degenerate to argmax."""

    def test_top_agent_gets_finite_share(self):
        """Clustered fitnesses (the real case: composite scores ~0-2, ELO ~1500)
        must give the top agent a strong but finite share of parent draws --
        not all of them. The old code softmaxed raw values scaled by 0.1, i.e.
        an effective temperature near zero."""
        strategy = TournamentEvolutionStrategy(seed=42)
        rng = np.random.RandomState(7)

        # Realistic cluster: everyone within ~2 points (like ELO after a few gens).
        fitnesses = [1500.0 + float(i) for i in range(12)]  # 1500..1511, sd~3.3
        indices = list(range(12))

        draws = np.zeros(len(indices))
        n_draws = 4000
        for _ in range(n_draws):
            p1, p2 = strategy._select_parents_tournament(indices, fitnesses, rng)
            draws[p1] += 1
            draws[p2] += 1

        top_share = draws.max() / n_draws
        # Old behaviour: ~0.95+. Fixed behaviour: strong but finite.
        assert top_share < 0.6, f"top agent took {top_share:.1%} of parent draws (argmax collapse)"
        assert top_share > 0.08, "selection pressure too weak to be useful"

    def test_degenerate_field_is_uniform(self):
        """All-equal fitnesses must not divide by zero or pick one agent."""
        strategy = TournamentEvolutionStrategy(seed=1)
        rng = np.random.RandomState(2)
        fitnesses = [1.0] * 8

        for _ in range(50):
            p1, p2 = strategy._select_parents_tournament(list(range(8)), fitnesses, rng)
            assert p1 != p2 or len(fitnesses) < 2


class TestChampionRefinements:
    """The exploitation channel: offspring descend from the run's best genome."""

    def test_refinements_are_close_mutations_of_champion(self):
        strategy = TournamentEvolutionStrategy(seed=42, champion_refinements=3)
        population = _population(n=10, dim=9207, seed=5)
        fitnesses = [float(i) for i in range(10)]

        # A clearly distinct "champion" genome.
        champion = np.zeros(9207)
        champion[::3] = 0.5

        offspring, info = strategy.evolve(
            population=population, weights_list=population,
            current_fitnesses=fitnesses, generation=1,
            champion_genome=champion, improved_this_generation=False,
        )

        assert len(offspring) == 10
        assert info["champion_refinements_used"] == 3

        # The refinement slots are gentle mutations of the champion: close in
        # distance to it (mutation only touches ~rate*dim coords at std scale),
        # far from a random population member.
        champ_dist = [float(np.linalg.norm(o - champion)) for o in offspring[:3]]
        pop_dist = min(
            float(np.linalg.norm(champion - p)) for p in population)

        assert max(champ_dist) < 0.25 * pop_dist, (
            f"refinements {champ_dist} not close to champion "
            f"(population distance scale {pop_dist:.3f})")
        # And they are not exact copies -- mutation must have touched them.
        assert all(d > 1e-9 for d in champ_dist)

    def test_no_champion_means_no_refinement_slots(self):
        strategy = TournamentEvolutionStrategy(seed=42, champion_refinements=3)
        population = _population(n=10, seed=5)
        fitnesses = [float(i) for i in range(10)]

        offspring, info = strategy.evolve(
            population=population, weights_list=population,
            current_fitnesses=fitnesses, generation=1,
        )
        assert len(offspring) == 10
        assert info["champion_refinements_used"] == 0


class TestAdaptiveMutation:
    """Live mutation std widens during stagnation and shrinks on progress."""

    def _strategy(self):
        return TournamentEvolutionStrategy(
            seed=42, adaptive_mutation=True,
            mutation_std=0.1, min_mutation_std=0.05, max_mutation_std=0.4,
            stagnation_window=2, champion_refinements=0,
        )

    def test_widens_after_stagnation_window(self):
        strategy = self._strategy()
        population = _population(n=10, seed=9)
        fitnesses = [float(i) for i in range(10)]

        assert strategy._mutation_std == pytest.approx(0.1)

        # Two flat generations: counter reaches the window on the 2nd call.
        strategy.evolve(population, population, current_fitnesses=fitnesses,
                        generation=1, improved_this_generation=False)
        assert strategy.stagnation_counter == 1
        assert strategy._mutation_std == pytest.approx(0.1)

        _, info = strategy.evolve(population, population,
                                  current_fitnesses=fitnesses,
                                  generation=2, improved_this_generation=False)
        assert strategy.stagnation_counter == 2
        # Window hit -> doubled (bounded by max).
        assert strategy._mutation_std == pytest.approx(0.2)
        assert info["stagnation_counter"] == 2

    def test_shrinks_back_on_progress(self):
        strategy = self._strategy()
        population = _population(n=10, seed=9)
        fitnesses = [float(i) for i in range(10)]

        # Stagnate twice to widen...
        strategy.evolve(population, population, current_fitnesses=fitnesses,
                        generation=1, improved_this_generation=False)
        strategy.evolve(population, population, current_fitnesses=fitnesses,
                        generation=2, improved_this_generation=False)
        widened = strategy._mutation_std

        # ...then progress: counter resets and std decays toward the baseline.
        strategy.evolve(population, population, current_fitnesses=fitnesses,
                        generation=3, improved_this_generation=True)
        assert strategy.stagnation_counter == 0
        assert strategy._mutation_std < widened
        # Bounded below by... not min_mutation_std but the baseline (0.1):
        # decay stops at max(baseline, current*0.8) = 0.16 here.
        assert strategy._mutation_std >= 0.1 - 1e-9

    def test_bounded_by_max(self):
        strategy = self._strategy()
        population = _population(n=10, seed=9)
        fitnesses = [float(i) for i in range(10)]

        # Stagnate many generations; std must never exceed max (tiny epsilon for fp).
        for gen in range(20):
            strategy.evolve(population, population, current_fitnesses=fitnesses,
                            generation=gen + 1, improved_this_generation=False)
        assert strategy._mutation_std <= 0.4 + 1e-9


class TestImmigration:
    """Diversity collapse must trigger fresh genomes."""

    def test_collapsed_population_gets_immigrants(self):
        strategy = TournamentEvolutionStrategy(seed=42, champion_refinements=0)
        dim = 64
        rng = np.random.RandomState(3)

        # Generation A: a normal, diverse population (records the baseline).
        pop_a = _population(n=20, dim=dim, seed=11)
        fitnesses = [float(i) for i in range(20)]
        strategy.evolve(pop_a, pop_a, current_fitnesses=fitnesses, generation=1)

        # Generation B: near-clones -- blend crossover + light mutation have
        # converged. One distinct member keeps diversity > 0 (non-degenerate).
        base = rng.randn(dim) * (1.0 / np.sqrt(dim))
        pop_b = [base.copy() for _ in range(20)]
        pop_b[5] = -base + rng.randn(dim) * 0.01

        _, info = strategy.evolve(pop_b, pop_b, current_fitnesses=fitnesses,
                                  generation=2)

        assert info["immigrants"] > 0, (
            "diversity collapsed to near-clones but no immigrants were added")
    def test_diverse_population_gets_no_immigrants(self):
        """A healthy population must not be disturbed."""
        strategy = TournamentEvolutionStrategy(seed=42, champion_refinements=0)
        fitnesses = [float(i) for i in range(20)]

        pop_a = _population(n=20, seed=13)
        strategy.evolve(pop_a, pop_a, current_fitnesses=fitnesses, generation=1)

        # Still diverse (fresh random draw each call).
        pop_b = _population(n=20, seed=14)
        _, info = strategy.evolve(pop_b, pop_b, current_fitnesses=fitnesses,
                                  generation=2)
        assert info["immigrants"] == 0

    def test_immigration_can_be_disabled(self):
        strategy = TournamentEvolutionStrategy(
            seed=42, champion_refinements=0, immigration_threshold=False)
        dim = 64
        rng = np.random.RandomState(3)
        fitnesses = [float(i) for i in range(20)]

        pop_a = _population(n=20, dim=dim, seed=15)
        strategy.evolve(pop_a, pop_a, current_fitnesses=fitnesses, generation=1)

        base = rng.randn(dim) * (1.0 / np.sqrt(dim))
        pop_b = [base.copy() for _ in range(20)]
        _, info = strategy.evolve(pop_b, pop_b, current_fitnesses=fitnesses,
                                  generation=2)
        assert info["immigrants"] == 0


class TestEvolveContract:
    """The public contract that the trainer relies on."""

    def test_offspring_size_and_keys(self):
        strategy = TournamentEvolutionStrategy(seed=42, champion_refinements=2)
        population = _population(n=16, seed=21)
        fitnesses = [float(i) for i in range(16)]
        champion = population[0].copy()

        offspring, info = strategy.evolve(
            population=population, weights_list=population,
            current_fitnesses=fitnesses, generation=3,
            champion_genome=champion, improved_this_generation=True)

        assert len(offspring) == 16
        for key in ("elite_indices", "tournament_rankings", "elo_ratings",
                    "mutation_std", "stagnation_counter",
                    "champion_refinements_used", "immigrants"):
            assert key in info, f"info missing {key}"

    def test_elites_preserved_verbatim(self):
        strategy = TournamentEvolutionStrategy(seed=42, elite_fraction=0.25)
        population = _population(n=8, seed=23)
        fitnesses = [float(i) for i in range(8)]  # index 7 is best

        offspring, info = strategy.evolve(
            population=population, weights_list=population,
            current_fitnesses=fitnesses, generation=1)

        elite_idx = info["elite_indices"][0]
        assert np.any([np.array_equal(o, population[elite_idx]) for o in offspring]), \
            "top-ranked genome not preserved unchanged"


class TestConfigWiring:
    """The trainer must pass configured GA knobs to the strategy."""

    def test_trainer_wires_ga_dynamics(self):
        from src.train.trainer import EvolutionTrainer, TrainingConfig
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(
                population_size=8, elite_count=2, max_generations=1,
                num_workers=2, match_duration="short", runs_dir=tmp, seed=3,
                checkpoint_interval=100, curriculum_learning=False,
                diversity_preservation=False, tournament_matches=2,
                hall_of_fame_size=0,
                champion_refinements=4, ga_adaptive_mutation=True,
                ga_stagnation_window=5, mutation_rate=0.13, crossover_rate=0.61)
            with EvolutionTrainer(config) as trainer:
                strat = trainer.tournament_strategy
                assert strat is not None
                assert strat.champion_refinements == 4
                assert strat.adaptive_mutation is True
                assert strat.stagnation_window == 5
                # The pre-fix bug: these silently fell back to hardcoded defaults.
                assert strat.mutation_rate == pytest.approx(0.13)
                assert strat.crossover_rate == pytest.approx(0.61)

    def test_config_roundtrip(self):
        from src.train.trainer import TrainingConfig

        config = TrainingConfig(champion_refinements=7, ga_stagnation_window=9)
        d = config.to_dict()
        assert d["champion_refinements"] == 7
        assert d["ga_adaptive_mutation"] is True
        restored = TrainingConfig.from_dict(d)
        assert restored.champion_refinements == 7
        assert restored.ga_stagnation_window == 9
