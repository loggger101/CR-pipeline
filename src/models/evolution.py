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
import math
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


# Short aliases accepted in configs alongside the enum member names.
_STRATEGY_ALIASES = {
    "single_point": "SINGLE_POINT",
    "tournament_elite": "TOURNAMENT_ELITE",
}


def _coerce_enum(value, enum_cls, field_name: str):
    """Accept an enum member or its name (any case) and return the member.

    Raises:
        ValueError: naming the field and the valid options, so a typo in a
            config surfaces immediately instead of silently selecting a
            default operator.
    """
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        key = _STRATEGY_ALIASES.get(value.lower(), value.upper())
        try:
            return enum_cls[key]
        except KeyError:
            pass
    valid = ", ".join(member.name.lower() for member in enum_cls)
    raise ValueError(
        f"{field_name}={value!r} is not a valid {enum_cls.__name__}; "
        f"expected one of: {valid}"
    )


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


def scale_mutation_rate(rate: float, genome_size: int,
                        max_expected_mutations: Optional[float]) -> float:
    """Cap a per-weight mutation rate by the expected mutations it implies.

    ``rate`` is a probability *per weight*, so the expected number of mutated
    coordinates in one offspring is ``rate * genome_size`` -- which grows with
    the network even when the configured rate does not change. Measured 2026-09:
    at rate 0.15, going from a 9,207-param to a 20,071-param policy pushed that
    expectation from ~1,381 to ~3,011 mutations/child, and the extra drift made
    selection *worse* (mean-fitness trend flipped +0.06..+0.11/gen to
    -0.10..+0.04/gen across seeds), while scaling the rate back down restored
    improvement (+0.09/+0.05/+0.10). This helper is what makes bigger genomes
    beneficial instead of silently degrading evolution: it leaves small/low-rate
    configs bit-identical (rate*size already under the cap) and only reduces a
    rate when ``rate * genome_size`` exceeds ``max_expected_mutations``.

    Args:
        rate: Configured per-weight mutation probability.
        genome_size: Number of weights in one offspring (0/None-safe).
        max_expected_mutations: Upper bound on expected mutations per offspring;
            None or <= 0 disables scaling entirely.

    Returns:
        ``min(rate, cap)`` where the cap keeps the expectation at or under the
        bound; unchanged when scaling is disabled or already within it.
    """
    if not max_expected_mutations or genome_size <= 0:
        return float(rate)
    capped = max(1.0 / genome_size, max_expected_mutations / genome_size)
    return min(float(rate), capped)


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
        max_expected_mutations: Cap on expected mutations per offspring
            (rate * genome_size). None/0 disables the cap; 1400 keeps a large
            policy at roughly one old-net's worth of drift -- see
            scale_mutation_rate for why this matters once genomes grow.
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
    # Expected-mutations cap (rate * genome_size). The default keeps the new
    # multi-layer policy's per-child drift at roughly one old-net's worth --
    # without it, doubling parameters more than doubled expected mutations and
    # selection got worse instead of better (see scale_mutation_rate).
    max_expected_mutations: Optional[float] = 1400.0
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
        # Callers (TrainingConfig, YAML configs) supply strategy names as
        # strings. Without coercion the enum comparisons in _init_selection
        # and friends never match, so every operator silently fell back to its
        # default -- a trainer asking for tournament selection got roulette --
        # and get_config_summary() raised AttributeError on `.name`, which
        # crashed checkpoint saving for any run long enough to reach one.
        self.selection_strategy = _coerce_enum(
            self.selection_strategy, SelectionStrategy, "selection_strategy")
        self.crossover_strategy = _coerce_enum(
            self.crossover_strategy, CrossoverStrategy, "crossover_strategy")
        self.mutation_strategy = _coerce_enum(
            self.mutation_strategy, MutationStrategy, "mutation_strategy")

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
        # Per-weight rate implies expected mutations = rate * genome_size, so a
        # bigger policy mutates more per child at an unchanged configured rate.
        # Cap it (no-op when already under the cap) -- see scale_mutation_rate.
        _genome_size = len(population[0]) if population else 0
        effective_mutate_rate = scale_mutation_rate(
            self.config.mutation_rate, _genome_size,
            self.config.max_expected_mutations)

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
            # entropy-seeded one, which left runs irreproducible. The rate is
            # genome-size-scaled (effective_mutate_rate) above.
            if self.config.adaptive_mutation:
                child1 = self.mutation.mutate(
                    child1, effective_mutate_rate,
                    rng=self.rng,
                    current_fitness=current_fitness,
                    avg_fitness=avg_fitness,
                )
                child2 = self.mutation.mutate(
                    child2, effective_mutate_rate,
                    rng=self.rng,
                    current_fitness=current_fitness,
                    avg_fitness=avg_fitness,
                )
            else:
                child1 = self.mutation.mutate(
                    child1, effective_mutate_rate,
                    self.config.mutation_std, rng=self.rng,
                )
                child2 = self.mutation.mutate(
                    child2, effective_mutate_rate,
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
        selection_temperature: float = 1.0,
        champion_refinements: int = 2,
        adaptive_mutation: bool = False,
        min_mutation_std: float = 0.01,
        max_mutation_std: float = 0.3,
        stagnation_window: int = 8,
        immigration_threshold: Optional[float] = None,
        max_expected_mutations: Optional[float] = 1400.0,
    ):
        """Initialize the tournament evolution strategy.

        Args:
            tournament_format: Format for the tournament
                ("round_robin", "single_elim", "double_elim", "league").
            matches_per_pair: Matches per pair in tournament.
            elite_fraction: Fraction of population that survives as elites.
            crossover_rate: Probability of crossover.
            mutation_rate: Per-weight mutation probability.
            mutation_std: Mutation noise standard deviation (baseline).
            seed: Random seed.
            rng: Random number generator.
            selection_temperature: Softmax temperature for parent selection over
                z-scored fitnesses. 1.0 gives the top agent a few-percent share
                of draws instead of ~all; higher = weaker pressure, lower =
                stronger. (The pre-2026-09 code softmaxed raw scores scaled by
                100x, i.e. temperature ~0.001: selection was effectively argmax
                and the population collapsed to clones of one lucky genome.)
            champion_refinements: Number of offspring per generation that are
                gentle mutations of the run's current best genome (exploitation
                channel -- this is what makes each generation build on the last).
            adaptive_mutation: Grow mutation std while no improvement lands,
                shrink it when progress resumes.
            min_mutation_std / max_mutation_std: Bounds for adaptive mutation.
            stagnation_window: Consecutive non-improving generations before the
                mutation step-up fires (adaptive mode only).
            immigration_threshold: If the population's mean pairwise distance
                drops below this, replace a few of the weakest slots with fresh
                random genomes to break clone collapse. None disables it.
            max_expected_mutations: Cap on expected mutations per offspring
                (mutation_rate * genome_size); see scale_mutation_rate for why
                an unscaled per-weight rate degrades evolution as genomes grow.
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

        # Advanced GA state. _mutation_std is the *live* value (adaptive mode
        # moves it); mutation_std stays as the configured baseline/floor anchor.
        self.selection_temperature = max(1e-3, float(selection_temperature))
        self.champion_refinements = max(0, int(champion_refinements))
        self.adaptive_mutation = adaptive_mutation
        self.min_mutation_std = min_mutation_std
        self.max_mutation_std = max(min_mutation_std, max_mutation_std)
        self.stagnation_window = max(1, int(stagnation_window))
        # None enables immigration with a *scale-free* baseline: the first
        # evolve call records the initial population's diversity and collapse is
        # flagged when later generations fall below 25% of it. A float value
        # pins an absolute mean-pairwise-distance threshold instead (useful if
        # you know your genome scale). False disables immigration entirely.
        self.immigration_threshold = immigration_threshold
        # Cap expected mutations per offspring (rate * genome_size); see
        # scale_mutation_rate for why an unscaled per-weight rate degrades a
        # larger policy's evolution instead of improving it.
        self.max_expected_mutations = max_expected_mutations
        self._baseline_diversity: float = 0.0
        self._mutation_std = float(mutation_std)
        self.stagnation_counter = 0

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
        champion_genome: Optional[np.ndarray] = None,
        improved_this_generation: bool = False,
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
            champion_genome: The run's best genome so far. When given, a few of
                the offspring are gentle mutations of it -- an exploitation
                channel that lets each generation build on the previous one
                instead of only recombining this generation's field.
            improved_this_generation: Whether the trainer recorded a new best
                since the last evolve call; drives adaptive mutation scaling.

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

        # Step 3a-prep: index-aligned scores for selection. ``tournament_rankings``
        # is sorted by rank (best first), so a plain [score, ...] list would hand
        # agent i the *i-th best* score instead of its own -- the old code was safe
        # only because it looked ratings up in a dict keyed by id. Rebuild from the
        # ranking's ids so selection sees each agent's true standing.
        _aid_to_score = {str(aid): float(score) for aid, score in tournament_rankings}
        tournament_fitnesses = [
            _aid_to_score.get(f"agent_{i}", 0.0) for i in range(n)]

        # Step 3a: Adaptive mutation scaling (only when enabled). While no new
        # best lands, hold; after ``stagnation_window`` flat generations double
        # the live std (bounded) so search widens instead of grinding in place.
        # Progress decays it back toward the configured baseline.
        if self.adaptive_mutation:
            if improved_this_generation:
                self.stagnation_counter = 0
                self._mutation_std = max(
                    self.mutation_std, self._mutation_std * 0.8)
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.stagnation_window:
                    self._mutation_std = min(
                        self.max_mutation_std, self._mutation_std * 2.0)

        # Step 3b: Create offspring using tournament-weighted selection over the
        # *z-scored* standings at a real temperature (see __init__). Parents are
        # drawn from the whole field, elites included -- excluding them preserved
        # the top performers as copies but barred them from passing on genes.
        offspring = []
        breeding_indices = list(range(n))

        slots_needed = n - len(elite_indices)

        # Exploitation channel: a few gentle mutations of the run's best genome,
        # so each generation can build directly on the previous one instead of
        # only recombining this generation's field. Half rate keeps them close.
        champion_slots = 0
        if champion_genome is not None and slots_needed > 0:
            champion_slots = min(self.champion_refinements, slots_needed)
            for _ in range(champion_slots):
                offspring.append(self._mutate(
                    np.array(champion_genome, copy=True),
                    rate=self.mutation_rate * 0.5))

        while len(offspring) < slots_needed:
            parent1_idx, parent2_idx = self._select_parents_tournament(
                breeding_indices, tournament_fitnesses, self.rng
            )

            p1 = population[parent1_idx]
            p2 = population[parent2_idx]

            # Crossover
            if self.rng.random() < self.crossover_rate:
                child1, child2 = self._crossover(p1, p2)
            else:
                child1 = p1.copy()
                child2 = p2.copy()

            # Mutation (live std -- adaptive mode may have widened it)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            offspring.append(child1)
            if len(offspring) < slots_needed:
                offspring.append(child2)

        # Step 3c: Immigration -- a scale-free collapse guard. The first call
        # records the initial (random) population's diversity as baseline; if a
        # later generation falls below a quarter of that, blend-crossover and
        # light mutation have converged to near-clones and selection pressure is
        # sorting noise. Replace a few slots with fresh random genomes so the
        # next tournament has something new to select on.
        immigrants = 0
        population_diversity = self._compute_diversity(population)
        if n >= 8:
            # Collapse threshold: an absolute float pins it; otherwise (None)
            # use a scale-free baseline recorded on the first call.
            collapse_below = None
            if isinstance(self.immigration_threshold, (int, float)):
                collapse_below = float(self.immigration_threshold)
            elif self._baseline_diversity <= 0.0:
                self._baseline_diversity = max(population_diversity, 1e-9)
            else:
                collapse_below = 0.25 * self._baseline_diversity

            if (collapse_below is not None
                    and population_diversity < collapse_below):
                k = min(max(1, n // 20), len(offspring) - champion_slots)
                if k > 0:
                    slots = list(range(champion_slots, len(offspring)))
                    for slot in self.rng.choice(slots, size=k, replace=False):
                        offspring[int(slot)] = self._random_genome(
                            len(offspring[0]))
                        immigrants += 1

        # Step 4: Add elites (preserve top tournament performers)
        for elite_idx in elite_indices:
            if len(offspring) < n:
                offspring.append(population[elite_idx].copy())

        # Trim to exact population size
        offspring = offspring[:n]

        info["mutation_std"] = self._mutation_std
        info["stagnation_counter"] = self.stagnation_counter
        info["champion_refinements_used"] = champion_slots
        info["immigrants"] = immigrants

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
            f"elite={len(elite_indices)}, champion_refinements={champion_slots}, "
            f"immigrants={immigrants}, mutation_std={self._mutation_std:.4f}, "
            f"diversity={diversity:.4f}"
        )

        return offspring, info

    def _select_parents_tournament(
        self,
        indices: List[int],
        fitnesses: List[float],
        rng: np.random.RandomState,
    ) -> Tuple[int, int]:
        """Select parents using tempered tournament-weighted selection.

        Fitness is z-scored first so the softmax operates on *relative* standing
        rather than raw magnitude -- raw scores (ELO ~1500 or composite fitness)
        differ by far less than their scale, and a fixed 0.1 scaling made the
        effective temperature near zero: one agent took ~all parent draws and
        the population collapsed to its clones within a few generations. With
        z-scored values at temperature ``selection_temperature``, the top agent
        gets a strong but finite share of draws and mid-field agents still breed.
        """
        if len(indices) < 2:
            return indices[0], indices[0]

        scores = np.array(
            [float(fitnesses[i]) for i in indices], dtype=float)
        # Guard a degenerate (all-equal) field: z-scoring would divide by zero.
        sd = float(scores.std())
        if not math.isfinite(sd) or sd < 1e-9:
            probs = np.full(len(indices), 1.0 / len(indices))
        else:
            z = (scores - scores.mean()) / sd
            # Temperature > 1 flattens the distribution; < 1 sharpens it.
            logits = z * self.selection_temperature
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()

        parent1 = rng.choice(indices, p=probs)
        parent2 = rng.choice(indices, p=probs)

        # Ensure different parents (same-parent "crossover" is just a copy).
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

    def _mutate(self, weights: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """Gaussian mutation using the *live* std (adaptive mode widens it).

        The per-weight rate is capped by genome size first (see
        scale_mutation_rate): an unscaled rate makes a bigger policy mutate more
        per child, which drowns selection instead of improving it.
        """
        base_rate = self.mutation_rate if rate is None else float(rate)
        effective_rate = scale_mutation_rate(
            base_rate, weights.shape[0] if weights.ndim > 0 else 0,
            self.max_expected_mutations)
        mutation_mask = self.rng.random(size=weights.shape) < effective_rate
        noise = self.rng.randn(*weights.shape) * self._mutation_std
        mutated = weights.copy()
        mutated[mutation_mask] += noise[mutation_mask]
        return mutated

    def _random_genome(self, size: int) -> np.ndarray:
        """Fresh genome for immigration.

        Per-coordinate scale 1/sqrt(size) mirrors the per-layer Xavier-style
        scaling of PolicySpec.random_genome (each weight ~N(0, 1/fan_in)); with
        no spec available here a uniform scale is close enough -- immigrants are
        exploration points, not calibrated agents.
        """
        return self.rng.randn(size) * np.sqrt(1.0 / max(1, size))

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
