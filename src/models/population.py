"""Population management for evolutionary training.

Handles storing, ranking, and managing a population of agents with
their fitness scores. Supports elite preservation, diversity tracking,
and population statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agent import EvolutionaryAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentRecord:
    """Record for a single agent in the population.

    Attributes:
        agent_id: Unique identifier.
        agent: The EvolutionaryAgent instance.
        fitness: Average fitness across evaluation matches.
        fitness_history: List of fitness scores across generations.
        rank: Current rank in the population (1 = best).
        wins: Total wins across all evaluations.
        losses: Total losses across all evaluations.
        metadata: Additional tracking info.
    """
    agent_id: str
    agent: EvolutionaryAgent
    fitness: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    rank: int = 0
    wins: int = 0
    losses: int = 0
    metadata: Dict = field(default_factory=dict)

    def __lt__(self, other: "AgentRecord") -> bool:
        """Compare by fitness (higher is better)."""
        return self.fitness > other.fitness


class Population:
    """Manages a population of evolutionary agents.

    Provides:
    - Population creation and initialization
    - Fitness ranking and sorting
    - Elite preservation
    - Diversity metrics
    - Population statistics
    - Checkpoint save/load
    """

    def __init__(self, population_size: int = 200,
                 elite_count: int = 10,
                 agent_config: Optional[dict] = None):
        """Initialize the population.

        Args:
            population_size: Number of agents in the population.
            elite_count: Number of elite agents to preserve.
            agent_config: Configuration dict for creating agents.
        """
        self.population_size = population_size
        self.elite_count = elite_count
        self.agent_config = agent_config or {}

        # Agent records
        self.agents: List[AgentRecord] = []
        self.generation: int = 0

        # History
        self.fitness_history: Dict[str, List[float]] = {
            "best": [],
            "mean": [],
            "median": [],
            "min": [],
            "std": [],
        }

    def initialize(self, seed: Optional[int] = None) -> None:
        """Initialize the population with random agents.

        Args:
            seed: Random seed for initialization.
        """
        rng = np.random.RandomState(seed)
        self.agents = []

        for i in range(self.population_size):
            # Create agent with random weights
            weights = rng.randn(1000) * 0.1  # Initial random weights
            agent = EvolutionaryAgent(
                weights=weights,
                seed=rng.randint(0, 2**31),
            )
            record = AgentRecord(
                agent_id=f"agent_{i}",
                agent=agent,
                fitness=0.0,
            )
            self.agents.append(record)

        self.generation = 0
        logger.info(f"Initialized population of {self.population_size} agents")

    def evaluate(self, fitness_scores: List[float]) -> None:
        """Update agent fitness scores and re-rank.

        Args:
            fitness_scores: List of fitness scores for each agent.
        """
        for i, score in enumerate(fitness_scores):
            if i < len(self.agents):
                self.agents[i].fitness = score
                self.agents[i].fitness_history.append(score)

        self._rank()
        self._update_history()

    def _rank(self) -> None:
        """Rank agents by fitness (highest first)."""
        sorted_agents = sorted(self.agents, key=lambda a: a.fitness, reverse=True)
        for rank, agent in enumerate(sorted_agents, 1):
            agent.rank = rank

    def _update_history(self) -> None:
        """Update population fitness history."""
        if not self.agents:
            return

        fitnesses = [a.fitness for a in self.agents]
        self.fitness_history["best"].append(max(fitnesses))
        self.fitness_history["mean"].append(np.mean(fitnesses))
        self.fitness_history["median"].append(np.median(fitnesses))
        self.fitness_history["min"].append(min(fitnesses))
        self.fitness_history["std"].append(np.std(fitnesses))

    def get_elite(self) -> List[AgentRecord]:
        """Get the elite agents (top N by fitness).

        Returns:
            List of elite AgentRecords.
        """
        return sorted(self.agents, key=lambda a: a.fitness, reverse=True)[:self.elite_count]

    def get_best(self) -> Optional[AgentRecord]:
        """Get the best agent in the population.

        Returns:
            Best AgentRecord, or None if population is empty.
        """
        if not self.agents:
            return None
        return max(self.agents, key=lambda a: a.fitness)

    def get_diversity(self) -> float:
        """Compute population diversity as mean pairwise weight distance.

        Returns:
            Average Euclidean distance between agent weight vectors.
        """
        if len(self.agents) < 2:
            return 0.0

        weights = [a.agent.get_weights() for a in self.agents]
        distances = []

        # Sample for efficiency
        sample_size = min(50, len(self.agents))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(self.agents), size=sample_size, replace=False)

        for i in indices:
            for j in indices:
                if i < j:
                    dist = np.linalg.norm(weights[i] - weights[j])
                    distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def get_statistics(self) -> Dict[str, float]:
        """Get population statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self.agents:
            return {}

        fitnesses = [a.fitness for a in self.agents]
        return {
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "mean_fitness": float(np.mean(fitnesses)),
            "median_fitness": float(np.median(fitnesses)),
            "min_fitness": min(fitnesses),
            "max_fitness": max(fitnesses),
            "std_fitness": float(np.std(fitnesses)),
            "diversity": self.get_diversity(),
            "elite_count": self.elite_count,
            "population_size": len(self.agents),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save population state to a file.

        Args:
            path: File path to save to.
        """
        import torch

        checkpoint = {
            "generation": self.generation,
            "fitness_history": self.fitness_history,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "agents": [],
        }

        for record in self.agents:
            checkpoint["agents"].append({
                "agent_id": record.agent_id,
                "weights": record.agent.get_weights(),
                "fitness": record.fitness,
                "fitness_history": record.fitness_history,
                "epsilon": record.agent.get_exploration_rate(),
            })

        torch.save(checkpoint, path)
        logger.info(f"Saved population checkpoint (gen {self.generation}) to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load population state from a file.

        Args:
            path: File path to load from.
        """
        import torch

        checkpoint = torch.load(path)
        self.generation = checkpoint["generation"]
        self.fitness_history = checkpoint["fitness_history"]
        self.population_size = checkpoint["population_size"]
        self.elite_count = checkpoint["elite_count"]

        self.agents = []
        for data in checkpoint["agents"]:
            agent = EvolutionaryAgent(weights=data["weights"])
            agent.epsilon = data.get("epsilon", 0.3)
            record = AgentRecord(
                agent_id=data["agent_id"],
                agent=agent,
                fitness=data["fitness"],
                fitness_history=data["fitness_history"],
            )
            self.agents.append(record)

        self._rank()
        logger.info(f"Loaded population checkpoint (gen {self.generation}) from {path}")

    def get_population_weights(self) -> List[np.ndarray]:
        """Get all agent weights as a list."""
        return [a.agent.get_weights() for a in self.agents]

    def set_population_weights(self, weights_list: List[np.ndarray]) -> None:
        """Set all agent weights from a list."""
        for i, w in enumerate(weights_list):
            if i < len(self.agents):
                self.agents[i].agent.set_weights(w)

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"Population(gen={self.generation}, size={len(self.agents)}, "
                f"best={stats.get('best_fitness', 0):.3f}, "
                f"mean={stats.get('mean_fitness', 0):.3f})")
