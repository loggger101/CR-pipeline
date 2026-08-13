"""Genetic algorithm operators for evolutionary training.

Implements:
- Selection strategies: tournament, rank-based, roulette wheel
- Crossover strategies: blend (BLX-α), single-point, uniform
- Mutation strategies: Gaussian, uniform, adaptive
- Elitism preservation
- Population replacement schemes
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Selection strategy enum."""
    TOURNAMENT = auto()
    RANK = auto()
    ROULETTE = auto()


class CrossoverStrategy(Enum):
    """Crossover strategy enum."""
    BLEND = auto()
    SINGLE_POINT = auto()
    UNIFORM = auto()


class MutationStrategy(Enum):
    """Mutation strategy enum."""
    GAUSSIAN = auto()
    UNIFORM = auto()
    ADAPTIVE = auto()


# ── Selection Operators ──────────────────────────────────────────────────────


class SelectionOperator(ABC):
    """Base class for selection operators."""

    @abstractmethod
    def select_parents(self, population: List, k: int = 2) -> Tuple[int, int]:
        """Select two parent indices from the population.

        Args:
            population: List of agents (or fitness scores).
            k: Tournament size (for tournament selection).

        Returns:
            Tuple of two parent indices.
        """
        pass


class TournamentSelection(SelectionOperator):
    """Tournament-based selection.

    Randomly selects k individuals and picks the fittest.
    Higher tournament size = stronger selection pressure.
    """

    def __init__(self, tournament_size: int = 5, rng: Optional[np.random.RandomState] = None):
        self.tournament_size = tournament_size
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float]) -> Tuple[int, int]:
        """Select two parents via tournament selection.

        Args:
            population: List of agents.
            fitnesses: List of fitness scores.

        Returns:
            Tuple of two parent indices.
        """
        parent1 = self._tournament(population, fitnesses)
        parent2 = self._tournament(population, fitnesses)
        while parent2 == parent1:
            parent2 = self._tournament(population, fitnesses)
        return parent1, parent2

    def _tournament(self, population: List, fitnesses: List[float]) -> int:
        """Run a single tournament."""
        candidates = self.rng.choice(len(population),
                                     size=min(self.tournament_size, len(population)),
                                     replace=False)
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

    def select_parents(self, population: List, fitnesses: List[float]) -> Tuple[int, int]:
        """Select two parents via rank-based selection."""
        # Rank individuals
        ranked_indices = np.argsort(fitnesses)[::-1]  # Descending

        # Compute selection probabilities based on rank
        n = len(population)
        probs = np.array([(n - rank) ** self.weight_exponent for rank in range(n)])
        probs /= probs.sum()

        parent1 = self.rng.choice(n, p=probs)
        parent2 = self.rng.choice(n, p=probs)
        while parent2 == parent1:
            parent2 = self.rng.choice(n, p=probs)

        return parent1, parent2


class RouletteSelection(SelectionOperator):
    """Roulette wheel selection.

    Selection probability proportional to fitness.
    Requires all fitness values to be positive.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def select_parents(self, population: List, fitnesses: List[float]) -> Tuple[int, int]:
        """Select two parents via roulette wheel."""
        # Shift fitness to be positive
        min_fit = min(fitnesses)
        shifted = [f - min_fit + 1e-6 for f in fitnesses]
        probs = np.array(shifted)
        probs /= probs.sum()

        parent1 = self.rng.choice(len(population), p=probs)
        parent2 = self.rng.choice(len(population), p=probs)
        while parent2 == parent1:
            parent2 = self.rng.choice(len(population), p=probs)

        return parent1, parent2


# ── Crossover Operators ──────────────────────────────────────────────────────


class CrossoverOperator(ABC):
    """Base class for crossover operators."""

    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents.

        Args:
            parent1: Weight vector of parent 1.
            parent2: Weight vector of parent 2.
            alpha: Crossover parameter.

        Returns:
            Tuple of two offspring weight vectors.
        """
        pass


class BlendCrossover(CrossoverOperator):
    """BLX-α blend crossover.

    Creates offspring by blending parent weights with noise.
    alpha=0.5 gives equal weighting; higher alpha = more exploration.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Perform blend crossover."""
        # Blend parents
        blend = self.rng.uniform(0, 1, size=parent1.shape)
        offspring1 = blend * parent1 + (1 - blend) * parent2
        offspring2 = blend * parent2 + (1 - blend) * parent1

        # Add small noise for diversity
        noise_scale = alpha * 0.1
        offspring1 += self.rng.randn(*parent1.shape) * noise_scale
        offspring2 += self.rng.randn(*parent2.shape) * noise_scale

        return offspring1, offspring2


class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover.

    Picks a random crossover point and swaps weights after that point.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Perform single-point crossover."""
        point = self.rng.randint(1, len(parent1))
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
                  alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover."""
        mask = self.rng.random(size=parent1.shape) > 0.5
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        return offspring1, offspring2


# ── Mutation Operators ───────────────────────────────────────────────────────


class MutationOperator(ABC):
    """Base class for mutation operators."""

    @abstractmethod
    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: float = 0.1) -> np.ndarray:
        """Mutate a weight vector.

        Args:
            weights: Weight vector to mutate.
            rate: Probability of mutation per weight.
            std: Standard deviation of mutation noise.

        Returns:
            Mutated weight vector.
        """
        pass


class GaussianMutation(MutationOperator):
    """Gaussian mutation.

    Adds Gaussian noise to each weight with probability `rate`.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng or np.random.RandomState()

    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: float = 0.1) -> np.ndarray:
        """Apply Gaussian mutation."""
        mutation_mask = self.rng.random(size=weights.shape) < rate
        noise = self.rng.randn(*weights.shape) * std
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
               std: float = 0.1) -> np.ndarray:
        """Apply uniform mutation."""
        mutation_mask = self.rng.random(size=weights.shape) < rate
        noise = self.rng.uniform(-std, std, size=weights.shape)
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
        self.current_std = std if 'std' in dir() else 0.1
        self.best_fitness = -float('inf')

    def mutate(self, weights: np.ndarray, rate: float = 0.05,
               std: Optional[float] = None, current_fitness: Optional[float] = None) -> np.ndarray:
        """Apply adaptive mutation."""
        if std is not None:
            self.current_std = std

        # Adjust mutation strength based on fitness improvement
        if current_fitness is not None:
            if current_fitness > self.best_fitness:
                self.current_std *= self.decrease_rate
                self.best_fitness = current_fitness
            else:
                self.current_std *= self.increase_rate

        # Clamp std
        self.current_std = max(self.min_std, min(self.max_std, self.current_std))

        # Apply Gaussian mutation with current std
        mutation_mask = self.rng.random(size=weights.shape) < rate
        noise = self.rng.randn(*weights.shape) * self.current_std
        mutated = weights.copy()
        mutated[mutation_mask] += noise[mutation_mask]
        return mutated


# ── Evolution Strategy ───────────────────────────────────────────────────────


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

    def __post_init__(self) -> None:
        if self.adaptive_mutation:
            self.mutation_strategy = MutationStrategy.ADAPTIVE


class EvolutionStrategy:
    """Orchestrates the genetic algorithm evolution loop.

    Manages selection, crossover, mutation, and elitism to produce
    the next generation from the current one.
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        """Initialize the evolution strategy.

        Args:
            config: Evolution configuration.
        """
        self.config = config or EvolutionConfig()
        self.rng = np.random.RandomState()

        # Initialize operators
        self._init_selection()
        self._init_crossover()
        self._init_mutation()

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
        else:
            self.selection = RouletteSelection(rng=self.rng)

    def _init_crossover(self) -> None:
        """Initialize the crossover operator."""
        if self.config.crossover_strategy == CrossoverStrategy.BLEND:
            self.crossover = BlendCrossover(rng=self.rng)
        elif self.config.crossover_strategy == CrossoverStrategy.SINGLE_POINT:
            self.crossover = SinglePointCrossover(rng=self.rng)
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

    def evolve(self, population: List, fitnesses: List[float],
               current_fitness: Optional[float] = None) -> List:
        """Evolve the population to produce the next generation.

        Args:
            population: Current population list.
            fitnesses: Fitness scores for each agent.
            current_fitness: Current generation fitness (for adaptive mutation).

        Returns:
            New population list (same size as input).
        """
        n = len(population)

        # Get elites
        if self.config.elite_preservation:
            elite_indices = np.argsort(fitnesses)[-self.config.elite_count:][::-1]
            elites = [(i, population[i]) for i in elite_indices]
        else:
            elites = []

        # Create offspring
        offspring = []
        while len(offspring) < n:
            # Select parents
            p1_idx, p2_idx = self.selection.select_parents(population, fitnesses)

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1, child2 = self.crossover.crossover(
                    population[p1_idx],
                    population[p2_idx],
                    alpha=self.config.blend_alpha,
                )
            else:
                child1 = population[p1_idx].copy()
                child2 = population[p2_idx].copy()

            # Mutation
            if self.config.adaptive_mutation:
                child1 = self.mutation.mutate(
                    child1, self.config.mutation_rate,
                    current_fitness=current_fitness,
                )
                child2 = self.mutation.mutate(
                    child2, self.config.mutation_rate,
                    current_fitness=current_fitness,
                )
            else:
                child1 = self.mutation.mutate(
                    child1, self.config.mutation_rate,
                    self.config.mutation_std,
                )
                child2 = self.mutation.mutate(
                    child2, self.config.mutation_rate,
                    self.config.mutation_std,
                )

            offspring.append(child1)
            if len(offspring) < n:
                offspring.append(child2)

        # Replace with elites
        if self.config.elite_preservation and elites:
            for i, (idx, _) in enumerate(elites):
                offspring[i] = population[idx].copy()

        return offspring

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
        }
