"""Neural Architecture Search for CR-Pipeline.

Provides:
- Search space definition for network architectures
- Evolutionary architecture search
- NAS-like architecture optimization
- Architecture performance prediction
- Architecture compression
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Architecture Components
# =============================================================================


class LayerType(Enum):
    """Supported layer types."""
    CONV2D = auto()
    CONV1D = auto()
    LSTM = auto()
    GRU = auto()
    DENSE = auto()
    BATCHNORM = auto()
    RELU = auto()
    LEAKY_RELU = auto()
    MAXPOOL = auto()
    AVGPOOL = auto()
    FLATTEN = auto()
    DROPOUT = auto()
    ATTENTION = auto()
    RESIDUAL = auto()


@dataclass
class ArchitectureConfig:
    """Configuration for a neural network architecture.

    Attributes:
        name: Architecture name.
        layers: List of layer configurations.
        input_shape: Input tensor shape.
        output_channels: Output channels.
        embedding_dim: Embedding dimension.
        num_heads: Attention heads.
        dropout_rate: Dropout rate.
        num_parameters: Estimated parameter count.
    """
    name: str
    layers: List[Dict[str, Any]] = field(default_factory=list)
    input_shape: Tuple[int, ...] = (8, 6, 16)
    output_channels: int = 64
    embedding_dim: int = 128
    num_heads: int = 4
    dropout_rate: float = 0.1
    num_parameters: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "layers": self.layers,
            "input_shape": list(self.input_shape),
            "output_channels": self.output_channels,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "num_parameters": self.num_parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureConfig":
        """Create from dictionary."""
        data["input_shape"] = tuple(data.get("input_shape", (8, 6, 16)))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# Architecture Search Space
# =============================================================================


class ArchitectureSearchSpace:
    """Defines the search space for architecture search.

    Contains configurable options for:
    - Layer types and counts
    - Filter sizes and channels
    - Activation functions
    - Pooling operations
    - Attention mechanisms
    """

    # Default search space
    DEFAULT_SPACE = {
        "conv_layers": {"min": 1, "max": 6, "step": 1},
        "conv_filters": {"min": 16, "max": 256, "step": 16},
        "conv_kernel_sizes": [3, 5, 7],
        "lstm_layers": {"min": 0, "max": 3, "step": 1},
        "lstm_units": {"min": 32, "max": 512, "step": 32},
        "dense_layers": {"min": 0, "max": 4, "step": 1},
        "dense_units": {"min": 32, "max": 512, "step": 32},
        "dropout_range": [0.0, 0.5],
        "use_attention": [True, False],
        "use_resnet": [True, False],
        "use_batchnorm": [True, False],
    }

    @classmethod
    def sample(cls, space: Optional[Dict[str, Any]] = None,
               rng: Optional[np.random.RandomState] = None) -> ArchitectureConfig:
        """Sample a random architecture from the search space.

        Args:
            space: Search space definition.
            rng: Random number generator.

        Returns:
            Sampled ArchitectureConfig.
        """
        s = space or cls.DEFAULT_SPACE
        r = rng or np.random.RandomState()

        n_conv = r.randint(s["conv_layers"]["min"], s["conv_layers"]["max"] + 1)
        n_lstm = r.randint(s["lstm_layers"]["min"], s["lstm_layers"]["max"] + 1)
        n_dense = r.randint(s["dense_layers"]["min"], s["dense_layers"]["max"] + 1)

        # Handle both dict and list formats for conv_filters
        if isinstance(s["conv_filters"], dict):
            step = s["conv_filters"].get("step", 16)
            # Generate values with step and pick one
            values = list(range(s["conv_filters"]["min"], s["conv_filters"]["max"] + 1, step))
            conv_filters = r.choice(values) if values else s["conv_filters"]["min"]
        else:
            conv_filters = r.choice(s["conv_filters"])

        kernel_size = r.choice(s["conv_kernel_sizes"])

        # Handle both dict and list formats for lstm_units
        if isinstance(s["lstm_units"], dict):
            step = s["lstm_units"].get("step", 32)
            values = list(range(s["lstm_units"]["min"], s["lstm_units"]["max"] + 1, step))
            lstm_units = r.choice(values) if values else s["lstm_units"]["min"]
        else:
            lstm_units = r.choice(s["lstm_units"])

        # Handle both dict and list formats for dense_units
        if isinstance(s["dense_units"], dict):
            step = s["dense_units"].get("step", 32)
            values = list(range(s["dense_units"]["min"], s["dense_units"]["max"] + 1, step))
            dense_units = r.choice(values) if values else s["dense_units"]["min"]
        else:
            dense_units = r.choice(s["dense_units"])
        dropout = r.uniform(*s["dropout_range"])
        use_attention = r.choice(s["use_attention"])
        use_resnet = r.choice(s["use_resnet"])
        use_bn = r.choice(s["use_batchnorm"])

        # Build layer list
        layers = []

        # Conv layers
        for i in range(n_conv):
            layers.append({
                "type": "conv2d",
                "filters": conv_filters * (2 ** i),
                "kernel_size": kernel_size,
                "activation": "relu",
                "batch_norm": use_bn,
            })
            if use_resnet and i < n_conv - 1:
                layers.append({"type": "residual"})

        # Pooling
        layers.append({"type": "maxpool", "pool_size": 2})

        # LSTM layers
        for i in range(n_lstm):
            layers.append({
                "type": "lstm",
                "units": lstm_units,
                "return_sequences": i < n_lstm - 1,
            })

        # Flatten
        layers.append({"type": "flatten"})

        # Dense layers
        for i in range(n_dense):
            units = dense_units // (2 ** i) if n_dense > 1 else dense_units
            layers.append({
                "type": "dense",
                "units": units,
                "activation": "relu" if i < n_dense - 1 else "linear",
                "dropout": dropout if i < n_dense - 1 else 0.0,
            })

        # Output
        layers.append({"type": "dense", "units": 5, "activation": "linear"})

        return ArchitectureConfig(
            name=f"arch_{int(time.time())}",
            layers=layers,
            dropout_rate=dropout,
            num_heads=4 if use_attention else 1,
        )

    @classmethod
    def get_default_space(cls) -> Dict[str, Any]:
        """Get the default search space."""
        return cls.DEFAULT_SPACE.copy()


# =============================================================================
# Architecture Evolution
# =============================================================================


class ArchitectureEvolver:
    """Evolves neural network architectures using genetic algorithms.

    Operations:
    - Mutation: Add/remove/modify layers
    - Crossover: Combine architectures
    - Selection: Based on performance
    """

    def __init__(
        self,
        search_space: Optional[Dict[str, Any]] = None,
        population_size: int = 20,
        elite_fraction: float = 0.2,
        seed: int = 42,
    ):
        """Initialize the architecture evolver.

        Args:
            search_space: Architecture search space.
            population_size: Population size.
            elite_fraction: Fraction of elites.
            seed: Random seed.
        """
        self.search_space = search_space or ArchitectureSearchSpace.get_default_space()
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.population: List[ArchitectureConfig] = []
        self.history: List[dict] = []

    def initialize_population(self) -> List[ArchitectureConfig]:
        """Initialize random population."""
        self.population = [
            ArchitectureSearchSpace.sample(self.search_space, self.rng)
            for _ in range(self.population_size)
        ]
        return self.population

    def evolve_generation(
        self,
        fitness_fn,
        n_generations: int = 50,
        verbose: bool = False,
    ) -> Tuple[List[ArchitectureConfig], dict]:
        """Evolve architectures for n generations.

        Args:
            fitness_fn: Function to evaluate architecture fitness.
            n_generations: Number of generations.
            verbose: Print progress.

        Returns:
            Tuple of (best_architecture, evolution_info).
        """
        if not self.population:
            self.initialize_population()

        best_arch = None
        best_fitness = -float("inf")
        info = {"fitness_history": [], "diversity_history": []}

        for gen in range(n_generations):
            # Evaluate fitness
            fitnesses = [fitness_fn(arch) for arch in self.population]

            # Track best
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_arch = self.population[max_idx]

            info["fitness_history"].append({
                "generation": gen,
                "best_fitness": float(best_fitness),
                "mean_fitness": float(np.mean(fitnesses)),
                "max_fitness": float(max(fitnesses)),
                "min_fitness": float(min(fitnesses)),
            })

            # Compute diversity
            diversity = self._compute_diversity()
            info["diversity_history"].append(diversity)

            if verbose and gen % 10 == 0:
                logger.info(f"Gen {gen}: best={best_fitness:.4f}, diversity={diversity:.4f}")

            # Select elites
            elite_count = max(1, int(self.population_size * self.elite_fraction))
            elite_indices = np.argsort(fitnesses)[-elite_count:][::-1]
            elites = [self.population[i] for i in elite_indices]
            elite_fitnesses = [fitnesses[i] for i in elite_indices]

            # Create next generation
            new_population = list(elites)

            while len(new_population) < self.population_size:
                # Tournament selection from elites
                p1, p2 = self._tournament_selection(elite_fitnesses)

                # Crossover
                child = self._crossover(elites[p1], elites[p2])

                # Mutation
                child = self._mutate(child)

                new_population.append(child)

            self.population = new_population

        info["best_architecture"] = best_arch.to_dict() if best_arch else None
        info["final_fitness"] = float(best_fitness)

        self.history.append(info)

        return [best_arch] if best_arch else [], info

    def _tournament_selection(self, fitnesses: List[float],
                               tournament_size: int = 3) -> Tuple[int, int]:
        """Select two parents via tournament."""
        pop_size = len(fitnesses)

        def select():
            k = min(tournament_size, pop_size)
            indices = self.rng.choice(pop_size, size=k, replace=False)
            return int(np.argmax([fitnesses[i] for i in indices]))

        p1 = select()
        p2 = select()
        while p2 == p1 and pop_size > 1:
            p2 = select()

        return p1, p2

    def _crossover(self, arch1: ArchitectureConfig,
                   arch2: ArchitectureConfig) -> ArchitectureConfig:
        """Crossover two architectures."""
        # Blend layer lists
        n_layers = max(len(arch1.layers), len(arch2.layers))
        blended_layers = []

        for i in range(n_layers):
            if i < len(arch1.layers) and i < len(arch2.layers):
                if self.rng.random() < 0.5:
                    blended_layers.append(arch1.layers[i])
                else:
                    blended_layers.append(arch2.layers[i])
            elif i < len(arch1.layers):
                blended_layers.append(arch1.layers[i])
            else:
                blended_layers.append(arch2.layers[i])

        # Blend hyperparameters
        config = ArchitectureConfig(
            name=f"arch_{int(time.time())}",
            layers=blended_layers,
            input_shape=arch1.input_shape,
            output_channels=max(arch1.output_channels, arch2.output_channels),
            embedding_dim=max(arch1.embedding_dim, arch2.embedding_dim),
            dropout_rate=(arch1.dropout_rate + arch2.dropout_rate) / 2,
        )

        return config

    def _mutate(self, arch: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture."""
        mutated = ArchitectureConfig(
            name=f"arch_{int(time.time())}",
            layers=[l.copy() for l in arch.layers],
            input_shape=arch.input_shape,
            output_channels=arch.output_channels,
            embedding_dim=arch.embedding_dim,
            dropout_rate=arch.dropout_rate,
        )

        # Random mutation
        if self.rng.random() < 0.3:
            # Add layer
            layer_types = ["conv2d", "dense", "dropout"]
            new_layer = {"type": self.rng.choice(layer_types)}
            pos = self.rng.randint(0, len(mutated.layers))
            mutated.layers.insert(pos, new_layer)

        if self.rng.random() < 0.2 and len(mutated.layers) > 2:
            # Remove layer
            pos = self.rng.randint(1, len(mutated.layers) - 1)
            mutated.layers.pop(pos)

        if self.rng.random() < 0.2:
            # Modify dropout
            mutated.dropout_rate = max(0.0, min(0.5,
                           mutated.dropout_rate + self.rng.normal(0, 0.05)))

        return mutated

    def _compute_diversity(self) -> float:
        """Compute population diversity."""
        if len(self.population) < 2:
            return 0.0

        total_dist = 0.0
        count = 0
        sample_size = min(10, len(self.population))
        indices = self.rng.choice(len(self.population), size=sample_size, replace=False)

        for i in indices:
            for j in indices:
                if i < j:
                    dist = self._architecture_distance(
                        self.population[i], self.population[j]
                    )
                    total_dist += dist
                    count += 1

        return float(total_dist / count) if count > 0 else 0.0

    def _architecture_distance(self, arch1: ArchitectureConfig,
                                arch2: ArchitectureConfig) -> float:
        """Compute distance between two architectures."""
        # Simple layer count distance
        layer_diff = abs(len(arch1.layers) - len(arch2.layers))

        # Parameter count distance (log scale)
        if arch1.num_parameters > 0 and arch2.num_parameters > 0:
            param_dist = abs(np.log1p(arch1.num_parameters) - np.log1p(arch2.num_parameters))
        else:
            param_dist = 0

        return layer_diff + param_dist


# =============================================================================
# Architecture Registry
# =============================================================================


class ArchitectureRegistry:
    """Registry for known architectures.

    Stores predefined and discovered architectures
    with performance tracking.
    """

    def __init__(self, registry_dir: str = "architecture_registry"):
        """Initialize the registry.

        Args:
            registry_dir: Directory for registry data.
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._architectures: Dict[str, ArchitectureConfig] = {}
        self._performance: Dict[str, List[float]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        reg_file = self.registry_dir / "registry.json"
        if reg_file.exists():
            try:
                with open(reg_file) as f:
                    data = json.load(f)
                for arch_id, arch_data in data.get("architectures", {}).items():
                    self._architectures[arch_id] = ArchitectureConfig.from_dict(arch_data)
                self._performance = data.get("performance", {})
            except (json.JSONDecodeError, IOError):
                pass

    def register_architecture(self, arch_id: str, config: ArchitectureConfig,
                               fitness: Optional[float] = None) -> None:
        """Register an architecture.

        Args:
            arch_id: Unique architecture ID.
            config: Architecture configuration.
            fitness: Fitness score.
        """
        self._architectures[arch_id] = config

        if fitness is not None:
            if arch_id not in self._performance:
                self._performance[arch_id] = []
            self._performance[arch_id].append(fitness)

        self._save_registry()

    def get_architecture(self, arch_id: str) -> Optional[ArchitectureConfig]:
        """Get an architecture by ID."""
        return self._architectures.get(arch_id)

    def get_best_architectures(self, n: int = 10,
                                metric: str = "best") -> List[Tuple[str, ArchitectureConfig, float]]:
        """Get top N architectures by performance.

        Args:
            n: Number of architectures to return.
            metric: Metric to rank by ("best", "mean", "last").

        Returns:
            List of (arch_id, config, score) tuples.
        """
        results = []

        for arch_id, scores in self._performance.items():
            if not scores:
                continue

            if metric == "best":
                score = max(scores)
            elif metric == "mean":
                score = float(np.mean(scores))
            else:
                score = scores[-1]

            arch = self._architectures.get(arch_id)
            if arch:
                results.append((arch_id, arch, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:n]

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        return {
            "total_architectures": len(self._architectures),
            "architectures_with_performance": len(self._performance),
            "best_architectures": [
                {"id": aid, "score": score}
                for aid, _, score in self.get_best_architectures(5)
            ],
        }

    def _save_registry(self) -> None:
        """Save registry to disk."""
        data = {
            "architectures": {
                aid: arch.to_dict() for aid, arch in self._architectures.items()
            },
            "performance": self._performance,
        }

        reg_file = self.registry_dir / "registry.json"
        with open(reg_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
