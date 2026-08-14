"""Population management for evolutionary training.

Handles storing, ranking, and managing a population of agents with
their fitness scores. Supports elite preservation, diversity tracking,
speciation, and population statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agent import EvolutionaryAgent
from .evolution import DiversityTracker
from .policy import DEFAULT_POLICY_SPEC, PolicySpec

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
        species_id: Species membership (for speciation).
        novelty_score: Novelty score (for novelty search).
        generation_born: Generation when this agent was created.
    """
    agent_id: str
    agent: EvolutionaryAgent
    # The evolved policy parameters (see models/policy.py). This, not the
    # agent's Torch network, is what selection/crossover/mutation operate on.
    genome: Optional[np.ndarray] = None
    fitness: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    rank: int = 0
    wins: int = 0
    losses: int = 0
    metadata: Dict = field(default_factory=dict)
    species_id: Optional[int] = None
    novelty_score: float = 0.0
    generation_born: int = 0

    def __lt__(self, other: "AgentRecord") -> bool:
        """Compare by fitness (higher is better)."""
        return self.fitness > other.fitness


class Population:
    """Manages a population of evolutionary agents.

    Provides:
    - Population creation and initialization
    - Fitness ranking and sorting
    - Elite preservation
    - Diversity metrics and tracking
    - Speciation support
    - Population statistics
    - Checkpoint save/load
    """

    def __init__(self, population_size: int = 200,
                 elite_count: int = 10,
                 agent_config: Optional[dict] = None,
                 policy_spec: Optional[PolicySpec] = None):
        """Initialize the population.

        Args:
            population_size: Number of agents in the population.
            elite_count: Number of elite agents to preserve.
            agent_config: Configuration dict for creating agents.
            policy_spec: Shape of the evolved policy. Defaults to
                ``DEFAULT_POLICY_SPEC``, which the parallel runner also uses;
                overriding it here without overriding it there will make
                genomes unusable at evaluation time.
        """
        self.population_size = population_size
        self.elite_count = elite_count
        self.agent_config = agent_config or {}
        self.policy_spec = policy_spec or DEFAULT_POLICY_SPEC

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

        # Diversity tracking
        self.diversity_tracker = DiversityTracker(sample_size=min(50, population_size))
        self.diversity_history: List[float] = []

        # Speciation
        self.species: Dict[int, List[int]] = {}  # species_id -> agent indices
        self.num_species: int = 0

    def initialize(self, seed: Optional[int] = None) -> None:
        """Initialize the population with random policy genomes.

        Genomes are sized and scaled for the policy in ``models/policy.py`` --
        the representation that actually decides what an agent plays. The
        previous code created an arbitrary 1000-element vector, stored it
        unused on the agent, and evolved the agent's 9.28M-parameter Torch
        network instead, which no code path ever consulted during a match.

        Args:
            seed: Random seed for initialization.
        """
        rng = np.random.RandomState(seed)
        self.agents = []

        for i in range(self.population_size):
            genome = self.policy_spec.random_genome(rng)
            agent = EvolutionaryAgent(
                weights=genome,
                seed=rng.randint(0, 2**31),
            )
            record = AgentRecord(
                agent_id=f"agent_{i}",
                agent=agent,
                genome=genome,
                fitness=0.0,
                generation_born=self.generation,
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
        self.fitness_history["mean"].append(float(np.mean(fitnesses)))
        self.fitness_history["median"].append(float(np.median(fitnesses)))
        self.fitness_history["min"].append(min(fitnesses))
        self.fitness_history["std"].append(float(np.std(fitnesses)))

    def update_diversity(self) -> float:
        """Update and return population diversity metric.

        Returns:
            Current diversity value.
        """
        weights = self.get_population_weights()
        diversity = self.diversity_tracker.compute_diversity(weights)
        self.diversity_history.append(diversity)
        self.diversity_tracker.record_diversity(diversity)
        return diversity

    def speciate(self, threshold: float = 0.5) -> Dict[int, List[int]]:
        """Perform speciation on the current population.

        Args:
            threshold: Maximum weight distance within a species.

        Returns:
            Dictionary mapping species_id to list of agent indices.
        """
        weights = self.get_population_weights()
        self.species = self.diversity_tracker.speciate(weights, threshold)
        self.num_species = len(self.species)
        
        # Assign species IDs to agents
        for species_id, member_indices in self.species.items():
            for idx in member_indices:
                if idx < len(self.agents):
                    self.agents[idx].species_id = species_id
        
        logger.info(f"Speciated population into {self.num_species} species")
        return self.species

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

        weights = self.get_population_weights()
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

    def get_species_diversity(self) -> Dict[int, float]:
        """Get diversity per species.

        Returns:
            Dictionary mapping species_id to diversity value.
        """
        if not self.species:
            return {}

        weights = self.get_population_weights()
        species_diversity = {}

        for species_id, member_indices in self.species.items():
            species_weights = [weights[idx] for idx in member_indices if idx < len(weights)]
            if len(species_weights) < 2:
                species_diversity[species_id] = 0.0
                continue

            distances = []
            for i in range(len(species_weights)):
                for j in range(i + 1, len(species_weights)):
                    distances.append(np.linalg.norm(species_weights[i] - species_weights[j]))
            species_diversity[species_id] = float(np.mean(distances)) if distances else 0.0

        return species_diversity

    def get_statistics(self) -> Dict[str, float]:
        """Get population statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self.agents:
            return {}

        fitnesses = [a.fitness for a in self.agents]
        stats = {
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
            "num_species": self.num_species,
        }

        # Add species stats if available
        if self.species:
            species_sizes = [len(members) for members in self.species.values()]
            stats["species_sizes"] = species_sizes
            stats["largest_species"] = max(species_sizes) if species_sizes else 0
            stats["smallest_species"] = min(species_sizes) if species_sizes else 0

        return stats

    def save_checkpoint(self, path: str) -> None:
        """Save population state to a file.

        Args:
            path: File path to save to.
        """
        import torch

        checkpoint = {
            "generation": self.generation,
            "fitness_history": self.fitness_history,
            "diversity_history": self.diversity_history,
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "num_species": self.num_species,
            "agents": [],
        }

        for record in self.agents:
            checkpoint["agents"].append({
                "agent_id": record.agent_id,
                "weights": np.asarray(record.genome),
                "fitness": record.fitness,
                "fitness_history": record.fitness_history,
                "epsilon": record.agent.get_exploration_rate(),
                "species_id": record.species_id,
                "novelty_score": record.novelty_score,
                "generation_born": record.generation_born,
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
        self.diversity_history = checkpoint.get("diversity_history", [])
        self.population_size = checkpoint["population_size"]
        self.elite_count = checkpoint["elite_count"]
        self.num_species = checkpoint.get("num_species", 0)

        self.agents = []
        for data in checkpoint["agents"]:
            genome = np.asarray(data["weights"])
            agent = EvolutionaryAgent(weights=genome)
            agent.epsilon = data.get("epsilon", 0.3)
            record = AgentRecord(
                agent_id=data["agent_id"],
                agent=agent,
                genome=genome,
                fitness=data["fitness"],
                fitness_history=data["fitness_history"],
                species_id=data.get("species_id"),
                novelty_score=data.get("novelty_score", 0.0),
                generation_born=data.get("generation_born", 0),
            )
            self.agents.append(record)

        self._rank()
        logger.info(f"Loaded population checkpoint (gen {self.generation}) from {path}")

    def get_population_weights(self) -> List[np.ndarray]:
        """Get every agent's policy genome, in population order."""
        return [np.asarray(a.genome) for a in self.agents]

    def set_population_weights(self, weights_list: List[np.ndarray]) -> None:
        """Replace agents' policy genomes (the output of an evolution step)."""
        for i, w in enumerate(weights_list):
            if i < len(self.agents):
                genome = np.asarray(w)
                self.agents[i].genome = genome
                # Keep the agent wrapper's view of its parameters in sync.
                self.agents[i].agent.weights = genome

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"Population(gen={self.generation}, size={len(self.agents)}, "
                f"best={stats.get('best_fitness', 0):.3f}, "
                f"mean={stats.get('mean_fitness', 0):.3f}, "
                f"species={self.num_species})")
