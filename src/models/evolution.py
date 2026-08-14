"""Genetic algorithm operators for evolutionary training.

Implements:
- Selection strategies: tournament, rank-based, roulette wheel, tournament_elite
- Crossover strategies: blend (BLX-alpha), single-point, uniform, arithmetic
- Mutation strategies: Gaussian, uniform, adaptive, speciated
- Elitism preservation with adaptive count
- Diversity tracking and preservation
- Speciation (NEAT-style) for niching
- Novelty search for behavior diversity
- Population management with fitness sharing
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..train import TournamentFormat

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class SelectionStrategy(Enum):
    """Selection strategy enum."""
    TOURNAMENT = auto()
    RANK = auto()
    ROULETTE = auto()
    TOURNAMENT_ELITE = auto()  # Tournament among elites


class CrossoverStrategy(Enum):
    """Crossover strategy enum."""
    BLEND = auto()
    SINGLE_POINT = auto()
    UNIFORM = auto()
    ARITHMETIC = auto()  # Weighted arithmetic mean


class MutationStrategy(Enum):
    """Mutation strategy enum."""
    GAUSSIAN = auto()
    UNIFORM = auto()
    ADAPTIVE = auto()
    SPECIATED = auto()  # Speciation-aware mutation


# =============================================================================
# Selection Operators
# =============================================================================

class SelectionOperator(ABC):
    """Base class for selection operators."""

    @abstractmethod
    def select_parents(self, population: List, fitnesses: List[float],
                       rng: np.random.RandomState) -> Tuple[int, int]:
        """Select two parent indices from the population."""
        pass


# Bound on re-draws when looking for a second, distinct parent.
_MAX_DISTINCT_ATTEMPTS = 32


def _resample_distinct(first: int, pop_size: int, draw) -> int:
    """Draw an index different from ``first``, giving up after a bounded try.

    A plain ``while second == first`` loop can never terminate: with
    ``tournament_size >= population_size`` every tournament sees the whole
    population and deterministically returns the same winner, so selection
    hangs. Small populations (the common case in tests and quick runs) hit
    this exactly. Falling back to a duplicate parent is harmless -- crossover
    of an individual with itself just reproduces it.
    """
    second = draw()
    if pop_size <= 1:
        return second
    for _ in range(_MAX_DISTINCT_ATTEMPTS):
        if second != first:
            return second
        second = draw()
    return second


class TournamentSelection(SelectionOperator):
    """Tournament-based selection.

    Randomly selects k individuals and picks the fittest.
    Higher tournament size = stronger selection pressure.
    """

    def __init__(self, tournament_size: int = 5, rng: Optional[np.random.RandomState] = None):
        self.tournament_size = tournament_size
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float],
                       rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        r = rng or self.rng
        n = len(population)
        parent1 = self._tournament(n, fitnesses, r)
        parent2 = _resample_distinct(
            parent1, n, lambda: self._tournament(n, fitnesses, r))
        return parent1, parent2

    def _tournament(self, pop_size: int, fitnesses: List[float],
                    rng: np.random.RandomState) -> int:
        k = min(self.tournament_size, pop_size)
        candidates = rng.choice(pop_size, size=k, replace=False)
        best = max(candidates, key=lambda i: fitnesses[i])
        return best


class RankSelection(SelectionOperator):
    """Rank-based selection.

    Ranks individuals by fitness and selects proportionally to rank.
    Reduces selection pressure compared to tournament selection.
    """

    def __init__(self, weight_exponent: float = 1.5,
                 rng: Optional[np.random.RandomState] = None):
        self.weight_exponent = weight_exponent
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float],
                       rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        r = rng or self.rng
        n = len(population)
        # Fittest first. `probs` is indexed by *rank*, so the draw must be
        # mapped back through ranked_indices to a population index. Returning
        # the raw draw (as this previously did) selected by array position
        # instead of by fitness -- which, for a population ordered worst-first,
        # meant rank selection preferentially bred the *worst* individuals.
        ranked_indices = np.argsort(fitnesses)[::-1]
        probs = np.array([(n - rank) ** self.weight_exponent for rank in range(n)],
                         dtype=np.float64)
        probs /= probs.sum()

        draw = lambda: int(ranked_indices[r.choice(n, p=probs)])
        parent1 = draw()
        return parent1, _resample_distinct(parent1, n, draw)


class RouletteSelection(SelectionOperator):
    """Roulette wheel selection.

    Selection probability proportional to fitness.
    Requires all fitness values to be positive (shifted if needed).
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float],
                       rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        r = rng or self.rng
        n = len(population)
        min_fit = min(fitnesses)
        shifted = np.array([f - min_fit + 1e-6 for f in fitnesses], dtype=np.float64)
        probs = shifted / shifted.sum()

        draw = lambda: int(r.choice(n, p=probs))
        parent1 = draw()
        return parent1, _resample_distinct(parent1, n, draw)


class TournamentEliteSelection(SelectionOperator):
    """Tournament selection with elite bias.

    First selects from elites with probability p_elite,
    then from the rest. Combines strong selection with diversity.
    """

    def __init__(self, tournament_size: int = 5, elite_fraction: float = 0.1,
                 rng: Optional[np.random.RandomState] = None):
        self.tournament_size = tournament_size
        self.elite_fraction = elite_fraction
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float],
                       rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        r = rng or self.rng
        n = len(population)
        elite_count = max(1, int(n * self.elite_fraction))

        draw = lambda: self._select(n, fitnesses, r, elite_count)
        parent1 = draw()
        return parent1, _resample_distinct(parent1, n, draw)

    def _select(self, n: int, fitnesses: List[float],
                rng: np.random.RandomState, elite_count: int) -> int:
        if rng.random() < 0.7:  # 70% chance to pick from elites
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            return int(rng.choice(elite_indices))
        k = min(self.tournament_size, n)
        candidates = rng.choice(n, size=k, replace=False)
        return int(max(candidates, key=lambda i: fitnesses[i]))


# =============================================================================
# Crossover Operators
# =============================================================================

class CrossoverOperator(ABC):
    """Base class for crossover operators."""

    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5, rng: Optional[np.random.RandomState] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class BlendCrossover(CrossoverOperator):
    """BLX-alpha blend crossover.

    Creates offspring by blending parent weights with noise.
    alpha=0.5 gives equal weighting; higher alpha = more exploration.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5,
                  rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        r = rng or self.rng
        blend = r.uniform(0, 1, size=parent1.shape)
        offspring1 = blend * parent1 + (1 - blend) * parent2
        offspring2 = blend * parent2 + (1 - blend) * parent1

        # Add small noise for diversity
        noise_scale = alpha * 0.1
        offspring1 += r.randn(*parent1.shape) * noise_scale
        offspring2 += r.randn(*parent2.shape) * noise_scale

        return offspring1, offspring2


class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover.

    Picks a random crossover point and swaps weights after that point.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5,
                  rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        r = rng or self.rng
        point = r.randint(1, len(parent1) - 1)
        offspring1 = np.concatenate([parent1[:point], parent2[point:]])
        offspring2 = np.concatenate([parent2[:point], parent1[point:]])
        return offspring1, offspring2


class UniformCrossover(CrossoverOperator):
    """Uniform crossover.

    For each weight, randomly choose which parent it comes from.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5,
                  rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        r = rng or self.rng
        mask = r.random(size=parent1.shape) > 0.5
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        return offspring1, offspring2


class ArithmeticCrossover(CrossoverOperator):
    """Arithmetic crossover with weighted combination.

    Creates offspring as weighted combinations of parents.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5,
                  rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        r = rng or self.rng
        # alpha controls the blend ratio
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = alpha * parent2 + (1 - alpha) * parent1
        return offspring1, offspring2


# =============================================================================
# Mutation Operators
# =============================================================================

class MutationOperator(ABC):
    """Base class for mutation operators."""

    @abstractmethod
    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: float = 0.1, rng: Optional[np.random.RandomState] = None,
               current_fitness: Optional[float] = None,
               avg_fitness: Optional[float] = None) -> np.ndarray:
        pass


class GaussianMutation(MutationOperator):
    """Gaussian mutation.

    Adds Gaussian noise to each weight with probability `rate`.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: float = 0.1,
               rng: Optional[np.random.RandomState] = None,
               current_fitness: Optional[float] = None,
               avg_fitness: Optional[float] = None) -> np.ndarray:
        r = rng or self.rng
        mutation_mask = r.random(size=weights.shape) < rate
        noise = r.randn(*weights.shape) * std
        mutated = weights.copy()
        mutated[mutation_mask] += noise[mutation_mask]
        return mutated


class UniformMutation(MutationOperator):
    """Uniform mutation.

    Replaces weights with uniform random values with probability `rate`.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: float = 0.1,
               rng: Optional[np.random.RandomState] = None,
               current_fitness: Optional[float] = None,
               avg_fitness: Optional[float] = None) -> np.ndarray:
        r = rng or self.rng
        mutation_mask = r.random(size=weights.shape) < rate
        noise = r.uniform(-std, std, size=weights.shape)
        mutated = weights.copy()
        mutated[mutation_mask] = noise[mutation_mask]
        return mutated


class AdaptiveMutation(MutationOperator):
    """Adaptive mutation with fitness-dependent std.

    Increases mutation strength when fitness plateaus, decreases
    when improving. Helps escape local optima.
    """

    def __init__(self, min_std: float = 0.01, max_std: float = 0.5,
                 increase_rate: float = 1.05, decrease_rate: float = 0.95,
                 rng: Optional[np.random.RandomState] = None):
        self.min_std = min_std
        self.max_std = max_std
        self.increase_rate = increase_rate
        self.decrease_rate = decrease_rate
        self.rng = rng or np.random.RandomState()
        self.current_std = 0.1
        self.best_fitness = -float('inf')
        self.no_improve_count = 0

    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: Optional[float] = None,
               rng: Optional[np.random.RandomState] = None,
               current_fitness: Optional[float] = None,
               avg_fitness: Optional[float] = None) -> np.ndarray:
        # `rng` sits in the same position as on every other mutation operator.
        # Omitting it here meant a positional call bound the RandomState to
        # current_fitness, which then blew up on a float comparison.
        r = rng or self.rng

        if std is not None:
            self.current_std = std

        # Adjust mutation strength based on fitness improvement
        if current_fitness is not None:
            if current_fitness > self.best_fitness + 1e-6:
                self.current_std *= self.decrease_rate
                self.best_fitness = current_fitness
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                # Increase mutation if no improvement for several generations
                if self.no_improve_count > 5:
                    self.current_std *= self.increase_rate

        # Clamp std
        self.current_std = max(self.min_std, min(self.max_std, self.current_std))

        # Apply Gaussian mutation
        mutation_mask = r.random(size=weights.shape) < rate
        noise = r.randn(*weights.shape) * self.current_std
        mutated = weights.copy()
        mutated[mutation_mask] += noise[mutation_mask]
        return mutated


# =============================================================================
# Diversity Tracking
# =============================================================================

class DiversityTracker:
    """Tracks and maintains population diversity.

    Provides:
    - Mean pairwise distance
    - Diversity preservation mechanisms
    - Speciation tracking
    """

    def __init__(self, sample_size: int = 50):
        self.sample_size = sample_size
        self.diversity_history: List[float] = []
        self.species: Dict[int, List[int]] = {}  # species_id -> agent indices
        self.next_species_id = 0

    def compute_diversity(self, weights_list: List[np.ndarray],
                          rng: Optional[np.random.RandomState] = None) -> float:
        """Compute population diversity as mean pairwise weight distance."""
        if len(weights_list) < 2:
            return 0.0

        r = rng or np.random.RandomState()
        sample_size = min(self.sample_size, len(weights_list))
        indices = r.choice(len(weights_list), size=sample_size, replace=False)

        total_dist = 0.0
        count = 0
        for i in indices:
            for j in indices:
                if i < j:
                    dist = np.linalg.norm(weights_list[i] - weights_list[j])
                    total_dist += dist
                    count += 1

        return float(total_dist / count) if count > 0 else 0.0

    def compute_pairwise_distances(self, weights_list: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise distances between all agents."""
        n = len(weights_list)
        if n < 2:
            return np.array([])

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weights_list[i] - weights_list[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances

    def speciate(self, weights_list: List[np.ndarray],
                 threshold: float = 0.5) -> Dict[int, List[int]]:
        """Perform simple UPGMA-style speciation based on weight distance.

        Args:
            weights_list: List of weight vectors.
            threshold: Maximum distance within a species.

        Returns:
            Dictionary mapping species_id to list of agent indices.
        """
        n = len(weights_list)
        if n == 0:
            return {}

        # Initialize: each agent in its own species
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Merge agents within threshold distance
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(weights_list[i] - weights_list[j])
                if dist < threshold:
                    union(i, j)

        # Group by species
        species = {}
        for i in range(n):
            root = find(i)
            if root not in species:
                species[root] = []
            species[root].append(i)

        # Reassign species IDs
        result = {}
        for new_id, (old_id, members) in enumerate(species.items()):
            result[new_id] = members
            self.next_species_id = new_id + 1

        self.species = result
        return result

    def record_diversity(self, diversity: float) -> None:
        """Record diversity value."""
        self.diversity_history.append(diversity)

    def get_diversity_stats(self) -> dict:
        """Get diversity statistics."""
        if not self.diversity_history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(self.diversity_history[-10:])),
            "std": float(np.std(self.diversity_history[-10:])),
            "min": float(min(self.diversity_history[-10:])),
            "max": float(max(self.diversity_history[-10:])),
        }


# =============================================================================
# Evolution Config
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary algorithm.

    Attributes:
        population_size: Number of agents per generation.
        elite_count: Number of elite agents preserved.
        elite_preservation: Whether to preserve elites.
        selection_strategy: Selection method.
        tournament_size: Tournament size (for tournament selection).
        rank_weight: Weight exponent for rank selection.
        crossover_strategy: Crossover method.
        crossover_rate: Probability of crossover.
        blend_alpha: Blend factor for blend crossover.
        mutation_strategy: Mutation method.
        mutation_rate: Per-weight mutation probability.
        mutation_std: Mutation noise standard deviation.
        min_mutation_std: Minimum mutation std (adaptive mode).
        max_mutation_std: Maximum mutation std (adaptive mode).
        adaptive_mutation: Whether to use adaptive mutation.
        diversity_preservation: Whether to maintain diversity.
        diversity_threshold: Threshold for speciation.
        novelty_search: Whether to use novelty search.
        novelty_window: Window for novelty tracking.
        seed: Seed for every stochastic operator. Leave None for
            non-deterministic behaviour; set it to make a run reproducible.
    """
    population_size: int = 200
    elite_count: int = 10
    elite_preservation: bool = True
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    tournament_size: int = 5
    rank_weight: float = 1.5
    crossover_strategy: CrossoverStrategy = CrossoverStrategy.BLEND
    crossover_rate: float = 0.7
    blend_alpha: float = 0.5
    mutation_strategy: MutationStrategy = MutationStrategy.GAUSSIAN
    mutation_rate: float = 0.05
    mutation_std: float = 0.1
    min_mutation_std: float = 0.01
    max_mutation_std: float = 0.5
    adaptive_mutation: bool = False
    diversity_preservation: bool = False
    diversity_threshold: float = 0.5
    novelty_search: bool = False
    novelty_window: int = 10
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.adaptive_mutation:
            self.mutation_strategy = MutationStrategy.ADAPTIVE


# =============================================================================
# Evolution Strategy
# =============================================================================

class EvolutionStrategy:
    """Orchestrates the genetic algorithm evolution loop.

    Manages selection, crossover, mutation, and elitism to produce
    the next generation from the current one.

    Enhanced with:
    - Diversity tracking and preservation
    - Speciation for niching
    - Adaptive mutation rates
    - Multiple selection/crossover/mutation strategies
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        # One seeded stream drives selection, crossover and mutation, so a
        # seeded config reproduces a run exactly. Previously this was an
        # entropy-seeded RandomState, which meant TrainingConfig.seed had no
        # effect on evolution at all.
        self.rng = np.random.RandomState(self.config.seed)

        # Initialize operators
        self._init_selection()
        self._init_crossover()
        self._init_mutation()

        # Diversity tracking
        self.diversity_tracker = DiversityTracker()
        self.novelty_archive: Dict[int, float] = {}  # agent_idx -> novelty score

    def _init_selection(self) -> None:
        """Initialize the selection operator."""
        if self.config.selection_strategy == SelectionStrategy.TOURNAMENT:
            self.selection = TournamentSelection(
                tournament_size=self.config.tournament_size,
                rng=self.rng,
            )
        elif self.config.selection_strategy == SelectionStrategy.RANK:
            self.selection = RankSelection(
                weight_exponent=self.config.rank_weight,
                rng=self.rng,
            )
        elif self.config.selection_strategy == SelectionStrategy.TOURNAMENT_ELITE:
            self.selection = TournamentEliteSelection(
                tournament_size=self.config.tournament_size,
                rng=self.rng,
            )
        else:
            self.selection = RouletteSelection(rng=self.rng)

    def _init_crossover(self) -> None:
        """Initialize the crossover operator."""
        if self.config.crossover_strategy == CrossoverStrategy.BLEND:
            self.crossover = BlendCrossover(rng=self.rng)
        elif self.config.crossover_strategy == CrossoverStrategy.SINGLE_POINT:
            self.crossover = SinglePointCrossover(rng=self.rng)
        elif self.config.crossover_strategy == CrossoverStrategy.ARITHMETIC:
            self.crossover = ArithmeticCrossover(rng=self.rng)
        else:
            self.crossover = UniformCrossover(rng=self.rng)

    def _init_mutation(self) -> None:
        """Initialize the mutation operator."""
        if self.config.mutation_strategy == MutationStrategy.GAUSSIAN:
            self.mutation = GaussianMutation(rng=self.rng)
        elif self.config.mutation_strategy == MutationStrategy.UNIFORM:
            self.mutation = UniformMutation(rng=self.rng)
        else:
            self.mutation = AdaptiveMutation(
                min_std=self.config.min_mutation_std,
                max_std=self.config.max_mutation_std,
                rng=self.rng,
            )

    def evolve(self, population: List[np.ndarray], fitnesses: List[float],
               current_fitness: Optional[float] = None,
               avg_fitness: Optional[float] = None) -> Tuple[List[np.ndarray], dict]:
        """Evolve the population to produce the next generation.

        Args:
            population: Current population list of weight arrays.
            fitnesses: Fitness scores for each agent.
            current_fitness: Current generation best fitness (for adaptive mutation).
            avg_fitness: Current generation average fitness.

        Returns:
            Tuple of (new_population, info_dict).
            info_dict contains diversity stats, speciation info, etc.
        """
        n = len(population)
        info = {}

        # Compute diversity
        if self.config.diversity_preservation:
            diversity = self.diversity_tracker.compute_diversity(population, self.rng)
            self.diversity_tracker.record_diversity(diversity)
            info["diversity"] = diversity

            # Speciate if needed
            if len(population) >= 4:
                speciation = self.diversity_tracker.speciate(population, self.config.diversity_threshold)
                info["speciation"] = speciation
                info["num_species"] = len(speciation)

        # Get elites
        if self.config.elite_preservation:
            elite_indices = np.argsort(fitnesses)[-self.config.elite_count:][::-1]
            elites = [(int(i), population[int(i)].copy()) for i in elite_indices]
        else:
            elites = []

        # Create offspring
        offspring = []
        while len(offspring) < n:
            # Select parents
            p1_idx, p2_idx = self.selection.select_parents(population, fitnesses, self.rng)

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1, child2 = self.crossover.crossover(
                    population[p1_idx],
                    population[p2_idx],
                    alpha=self.config.blend_alpha,
                    rng=self.rng,
                )
            else:
                child1 = population[p1_idx].copy()
                child2 = population[p2_idx].copy()

            # Mutation. rng is passed explicitly so mutation draws from the
            # strategy's seeded stream rather than the operator's own
            # entropy-seeded one, which left runs irreproducible.
            if self.config.adaptive_mutation:
                child1 = self.mutation.mutate(
                    child1, self.config.mutation_rate,
                    rng=self.rng,
                    current_fitness=current_fitness,
                    avg_fitness=avg_fitness,
                )
                child2 = self.mutation.mutate(
                    child2, self.config.mutation_rate,
                    rng=self.rng,
                    current_fitness=current_fitness,
                    avg_fitness=avg_fitness,
                )
            else:
                child1 = self.mutation.mutate(
                    child1, self.config.mutation_rate,
                    self.config.mutation_std, rng=self.rng,
                )
                child2 = self.mutation.mutate(
                    child2, self.config.mutation_rate,
                    self.config.mutation_std, rng=self.rng,
                )

            offspring.append(child1)
            if len(offspring) < n:
                offspring.append(child2)

        # Replace with elites
        if self.config.elite_preservation and elites:
            for i, (idx, _) in enumerate(elites):
                if i < len(offspring):
                    offspring[i] = population[idx].copy()

        # Compute diversity of offspring
        if self.config.diversity_preservation:
            offspring_diversity = self.diversity_tracker.compute_diversity(offspring, self.rng)
            info["offspring_diversity"] = offspring_diversity

        return offspring, info

    def get_config_summary(self) -> dict:
        """Get a summary of the evolution configuration."""
        return {
            "selection": self.config.selection_strategy.name,
            "crossover": self.config.crossover_strategy.name,
            "mutation": self.config.mutation_strategy.name,
            "crossover_rate": self.config.crossover_rate,
            "mutation_rate": self.config.mutation_rate,
            "mutation_std": self.config.mutation_std,
            "elite_count": self.config.elite_count,
            "elite_preservation": self.config.elite_preservation,
            "diversity_preservation": self.config.diversity_preservation,
            "novelty_search": self.config.novelty_search,
            "diversity_stats": self.diversity_tracker.get_diversity_stats(),
        }


# =============================================================================
# Tournament-Based Evolution Strategy
# =============================================================================


class TournamentEvolutionStrategy:
    """Evolution strategy driven by tournament rankings.

    Instead of using raw fitness scores for selection, this strategy
    runs a tournament each generation to rank the population.
    Tournament rankings drive selection pressure, elitism, and
    crossover/mutation decisions.

    Features:
    - Tournament-based ranking each generation
    - Tournament-driven elitism (top performers survive)
    - Seeded matchups (higher-ranked agents face weaker opponents)
    - ELO-based selection pressure
    - Tournament history tracking for analysis
    """

    def __init__(
        self,
        tournament_format: str = "round_robin",
        matches_per_pair: int = 4,
        elite_fraction: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.05,
        mutation_std: float = 0.1,
        seed: int = 42,
        rng: Optional[np.random.RandomState] = None,
    ):
        """Initialize the tournament evolution strategy.

        Args:
            tournament_format: Format for the tournament
                ("round_robin", "single_elim", "double_elim", "league").
            matches_per_pair: Matches per pair in tournament.
            elite_fraction: Fraction of population that survives as elites.
            crossover_rate: Probability of crossover.
            mutation_rate: Per-weight mutation probability.
            mutation_std: Mutation noise standard deviation.
            seed: Random seed.
            rng: Random number generator.
        """
        # Convert string format to enum
        format_map = {
            "round_robin": "ROUND_ROBIN",
            "single_elim": "SINGLE_ELIMINATION",
            "double_elim": "DOUBLE_ELIMINATION",
            "league": "LEAGUE",
        }
        format_name = format_map.get(tournament_format, "ROUND_ROBIN")
        self.tournament_format = format_name
        self.matches_per_pair = matches_per_pair
        self.elite_fraction = elite_fraction
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.seed = seed
        self.rng = rng or np.random.RandomState(seed)

        # Tournament history
        self.tournament_history: List[dict] = []
        self.elo_history: Dict[str, List[float]] = {}

    def evolve(
        self,
        population: List[np.ndarray],
        weights_list: List[np.ndarray],
        current_fitnesses: Optional[List[float]] = None,
        evaluator: Optional[Any] = None,
        generation: int = 0,
    ) -> Tuple[List[np.ndarray], dict]:
        """Evolve the population using tournament-based selection.

        Runs a tournament to rank agents, then uses tournament rankings
        to drive selection, crossover, and mutation.

        Args:
            population: Current population weight arrays.
            weights_list: Same as population (for compatibility).
            current_fitnesses: Current fitness scores (fallback if no evaluator).
            evaluator: FitnessEvaluator for running tournaments.
            generation: Current generation number.

        Returns:
            Tuple of (new_population, info_dict).
        """
        n = len(population)
        info = {}

        # Step 1: Run tournament to get rankings
        # Pre-calculate elite count for use in both branches
        elite_count = max(1, int(n * self.elite_fraction))

        if evaluator is not None:
            agent_ids = [f"agent_{i}" for i in range(n)]
            tournament_result = evaluator.run_tournament(
                agent_ids=agent_ids,
                weights_list=weights_list,
                format=self.tournament_format,
                matches_per_pair=self.matches_per_pair,
                seed=self.seed + generation * 1000,
                generation=generation,
            )
            # Get tournament rankings
            tournament_rankings = tournament_result.rankings
            elo_ratings = tournament_result.elo_ratings
            h2h_records = tournament_result.h2h_records

            # Convert rankings to fitness-like scores
            tournament_fitnesses = [score for _, score in tournament_rankings]
        else:
            # Fallback: use raw fitnesses if no evaluator
            if current_fitnesses is not None:
                tournament_fitnesses = current_fitnesses
                elo_ratings = {f"agent_{i}": 1500.0 + f * 100 for i, f in enumerate(current_fitnesses)}
            else:
                tournament_fitnesses = [0.0] * n
                elo_ratings = {f"agent_{i}": 1500.0 for i in range(n)}

            # Use fitness-based ranking
            ranking_indices = np.argsort(tournament_fitnesses)[::-1]
            tournament_rankings = [
                (f"agent_{i}", tournament_fitnesses[int(i)]) for i in ranking_indices
            ]
            h2h_records = {}
            # Store integer indices for elite tracking
            elite_candidate_indices = ranking_indices[:elite_count].tolist()

        # Record ELO history
        for aid, elo in elo_ratings.items():
            if aid not in self.elo_history:
                self.elo_history[aid] = []
            self.elo_history[aid].append(elo)

        # Step 2: Select elites based on tournament ranking
        # Extract integer indices for elite tracking
        if evaluator is not None:
            # When using evaluator, map agent IDs back to indices
            elite_indices = [int(aid.replace("agent_", "")) for aid, _ in tournament_rankings[:elite_count]]
        else:
            # Fallback: use pre-computed integer indices
            elite_indices = [int(i) for i in elite_candidate_indices]
        info["elite_indices"] = elite_indices
        info["tournament_rankings"] = tournament_rankings
        info["elo_ratings"] = elo_ratings

        # Step 3: Create offspring using tournament-weighted selection.
        # Parents are drawn from the *whole* field, elites included. Excluding
        # elites (as this previously did) preserved the top performers as
        # copies but barred them from passing on any genes, so every new
        # genome descended only from agents the tournament had just ranked
        # below them.
        offspring = []
        breeding_indices = list(range(n))

        while len(offspring) < n - len(elite_indices):
            parent1_idx, parent2_idx = self._select_parents_tournament(
                breeding_indices, elo_ratings, self.rng
            )

            p1 = population[parent1_idx]
            p2 = population[parent2_idx]

            # Crossover
            if self.rng.random() < self.crossover_rate:
                child1, child2 = self._crossover(p1, p2)
            else:
                child1 = p1.copy()
                child2 = p2.copy()

            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            offspring.append(child1)
            if len(offspring) < n - len(elite_indices):
                offspring.append(child2)

        # Step 4: Add elites (preserve top tournament performers)
        for elite_idx in elite_indices:
            if len(offspring) < n:
                offspring.append(population[elite_idx].copy())

        # Trim to exact population size
        offspring = offspring[:n]

        # Step 5: Record tournament info
        info["tournament_history"] = {
            "generation": generation,
            "rankings": [(aid, float(score)) for aid, score in tournament_rankings],
            "elite_indices": elite_indices,
            "elo_ratings": {aid: float(elo) for aid, elo in elo_ratings.items()},
        }
        self.tournament_history.append(info["tournament_history"])

        # Compute diversity of offspring
        diversity = self._compute_diversity(offspring)
        info["offspring_diversity"] = diversity

        logger.info(
            f"Tournament evolution gen {generation}: "
            f"elite={elite_count}, diversity={diversity:.4f}"
        )

        return offspring, info

    def _select_parents_tournament(
        self,
        indices: List[int],
        elo_ratings: Dict[str, float],
        rng: np.random.RandomState,
    ) -> Tuple[int, int]:
        """Select parents using tournament-weighted selection.

        Higher ELO ratings have higher selection probability.
        """
        if len(indices) < 2:
            return indices[0], indices[0]

        # Convert ELO to selection probabilities
        elos = np.array([elo_ratings.get(f"agent_{i}", 1500.0) for i in indices], dtype=float)
        # Softmax for probabilities
        elos_shifted = elos - np.max(elos)
        probs = np.exp(elos_shifted * 0.1)  # Scale ELO for numerical stability
        probs /= probs.sum()

        parent1 = rng.choice(indices, p=probs)
        parent2 = rng.choice(indices, p=probs)

        # Ensure different parents
        attempts = 0
        while parent2 == parent1 and attempts < 10:
            parent2 = rng.choice(indices, p=probs)
            attempts += 1

        return parent1, parent2

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Blend crossover with noise."""
        blend = self.rng.uniform(0, 1, size=parent1.shape)
        offspring1 = blend * parent1 + (1 - blend) * parent2
        offspring2 = blend * parent2 + (1 - blend) * parent1

        # Add small noise for diversity
        offspring1 += self.rng.randn(*parent1.shape) * 0.01
        offspring2 += self.rng.randn(*parent2.shape) * 0.01

        return offspring1, offspring2

    def _mutate(self, weights: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        mutation_mask = self.rng.random(size=weights.shape) < self.mutation_rate
        noise = self.rng.randn(*weights.shape) * self.mutation_std
        mutated = weights.copy()
        mutated[mutation_mask] += noise[mutation_mask]
        return mutated

    def _compute_diversity(self, weights_list: List[np.ndarray]) -> float:
        """Compute population diversity as mean pairwise distance."""
        if len(weights_list) < 2:
            return 0.0

        sample_size = min(20, len(weights_list))
        indices = self.rng.choice(len(weights_list), size=sample_size, replace=False)

        total_dist = 0.0
        count = 0
        for i in indices:
            for j in indices:
                if i < j:
                    dist = np.linalg.norm(weights_list[i] - weights_list[j])
                    total_dist += dist
                    count += 1

        return float(total_dist / count) if count > 0 else 0.0

    def get_tournament_summary(self, last_n: int = 10) -> dict:
        """Get a summary of recent tournament results."""
        recent = self.tournament_history[-last_n:]
        if not recent:
            return {}

        summary = {
            "generations": [h["generation"] for h in recent],
            "elite_indices": [h["elite_indices"] for h in recent],
        }

        # Track top agent ELO over time
        if recent:
            top_elo = []
            for h in recent:
                rankings = h["rankings"]
                if rankings:
                    top_elo.append(rankings[0][1])  # Top score
            summary["top_score_history"] = top_elo

        return summary


# =============================================================================
# Tournament-aware selection operators (for use with EvolutionStrategy)
# =============================================================================


class TournamentRankSelection(SelectionOperator):
    """Selection based on tournament rank (not raw fitness).

    Ranks individuals by tournament score and selects proportionally.
    Provides stronger selection pressure than fitness-based selection.
    """

    def __init__(self, weight_exponent: float = 2.0,
                 rng: Optional[np.random.RandomState] = None):
        self.weight_exponent = weight_exponent
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float],
                       rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        r = rng or self.rng
        n = len(population)

        # Sort by fitness to get ranks
        ranked_indices = np.argsort(fitnesses)[::-1]  # Descending

        # Compute selection probabilities based on rank
        probs = np.array([(n - rank) ** self.weight_exponent for rank in range(n)])
        probs /= probs.sum()

        parent1 = r.choice(n, p=probs)
        parent2 = r.choice(n, p=probs)
        while parent2 == parent1 and n > 1:
            parent2 = r.choice(n, p=probs)
        return parent1, parent2


class TournamentEliteSelection(SelectionOperator):
    """Tournament selection with elite bias.

    First selects from elites with probability p_elite,
    then from the rest. Combines strong selection with diversity.
    """

    def __init__(self, tournament_size: int = 5, elite_fraction: float = 0.1,
                 rng: Optional[np.random.RandomState] = None):
        self.tournament_size = tournament_size
        self.elite_fraction = elite_fraction
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float],
                       rng: Optional[np.random.RandomState] = None) -> Tuple[int, int]:
        r = rng or self.rng
        n = len(population)
        elite_count = max(1, int(n * self.elite_fraction))

        parent1 = self._select(n, fitnesses, r, elite_count)
        parent2 = self._select(n, fitnesses, r, elite_count)
        while parent2 == parent1 and n > 1:
            parent2 = self._select(n, fitnesses, r, elite_count)
        return parent1, parent2

    def _select(self, n: int, fitnesses: List[float],
                rng: np.random.RandomState, elite_count: int) -> int:
        if rng.random() < 0.7:  # 70% chance to pick from elites
            elite_indices = np.argsort(fitnesses)[-elite_count:]
            return rng.choice(elite_indices)
        else:
            k = min(self.tournament_size, n)
            candidates = rng.choice(n, size=k, replace=False)
            return max(candidates, key=lambda i: fitnesses[i])
