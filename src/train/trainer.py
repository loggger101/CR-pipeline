"""Main evolution trainer for CR-Pipeline.

Orchestrates the full evolutionary training loop:
1. Initialize population
2. For each generation:
   a. Evaluate fitness (parallel simulation)
   b. Apply evolution operators
   c. Log metrics and checkpoint
3. Handle early stopping and resume

Configurable via YAML config files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..models import (
    Population, AgentRecord, EvolutionStrategy, EvolutionConfig,
)
from ..env.sim import ParallelRunner, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline.

    Attributes:
        population_size: Number of agents per generation.
        elite_count: Number of elite agents to preserve.
        elite_preservation: Whether to preserve elites.
        selection_strategy: Selection method name.
        tournament_size: Tournament size.
        rank_weight: Weight exponent for rank selection.
        crossover_strategy: Crossover method name.
        crossover_rate: Probability of crossover.
        blend_alpha: Blend factor.
        mutation_strategy: Mutation method name.
        mutation_rate: Per-weight mutation probability.
        mutation_std: Mutation noise standard deviation.
        min_mutation_std: Minimum mutation std.
        max_mutation_std: Maximum mutation std.
        adaptive_mutation: Whether to use adaptive mutation.
        matches_per_agent: Matches per agent per generation.
        match_duration: Match duration ("full", "short", "overtime").
        scoring_weights: Dict of fitness scoring weights.
        max_generations: Maximum training generations.
        early_stopping_patience: Generations without improvement to stop.
        early_stopping_min_improvement: Minimum improvement threshold.
        checkpoint_interval: Save checkpoint every N generations.
        max_checkpoints: Maximum checkpoints to keep.
        num_workers: Number of parallel workers.
        batch_size: Agents per evaluation batch.
        timeout: Max seconds per evaluation batch.
        log_interval: Log every N generations.
        save_full_history: Whether to save full fitness history.
        runs_dir: Directory for training outputs.
        resume_from: Path to resume from checkpoint.
        seed: Random seed.
    """
    population_size: int = 200
    elite_count: int = 10
    elite_preservation: bool = True
    selection_strategy: str = "tournament"
    tournament_size: int = 5
    rank_weight: float = 1.5
    crossover_strategy: str = "blend"
    crossover_rate: float = 0.7
    blend_alpha: float = 0.5
    mutation_strategy: str = "gaussian"
    mutation_rate: float = 0.05
    mutation_std: float = 0.1
    min_mutation_std: float = 0.01
    max_mutation_std: float = 0.5
    adaptive_mutation: bool = False
    matches_per_agent: int = 5
    match_duration: str = "full"
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "trophy_gain_weight": 0.4,
        "towers_destroyed_weight": 0.3,
        "win_bonus": 0.2,
        "efficiency_weight": 0.1,
    })
    max_generations: int = 500
    early_stopping_patience: int = 30
    early_stopping_min_improvement: float = 0.5
    checkpoint_interval: int = 10
    max_checkpoints: int = 50
    num_workers: int = 4
    batch_size: int = 50
    timeout: int = 300
    log_interval: int = 1
    save_full_history: bool = True
    runs_dir: str = "runs"
    resume_from: Optional[str] = None
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "population_size": self.population_size,
            "elite_count": self.elite_count,
            "elite_preservation": self.elite_preservation,
            "selection_strategy": self.selection_strategy,
            "tournament_size": self.tournament_size,
            "rank_weight": self.rank_weight,
            "crossover_strategy": self.crossover_strategy,
            "crossover_rate": self.crossover_rate,
            "blend_alpha": self.blend_alpha,
            "mutation_strategy": self.mutation_strategy,
            "mutation_rate": self.mutation_rate,
            "mutation_std": self.mutation_std,
            "min_mutation_std": self.min_mutation_std,
            "max_mutation_std": self.max_mutation_std,
            "adaptive_mutation": self.adaptive_mutation,
            "matches_per_agent": self.matches_per_agent,
            "match_duration": self.match_duration,
            "scoring_weights": self.scoring_weights,
            "max_generations": self.max_generations,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_improvement": self.early_stopping_min_improvement,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoints": self.max_checkpoints,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "log_interval": self.log_interval,
            "save_full_history": self.save_full_history,
            "runs_dir": self.runs_dir,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class EvolutionTrainer:
    """Main evolution training loop orchestrator.

    Manages the full lifecycle of evolutionary training:
    - Population initialization
    - Fitness evaluation via parallel simulation
    - Evolution operations
    - Checkpointing
    - Early stopping
    - Metrics logging
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.population = Population(
            population_size=config.population_size,
            elite_count=config.elite_count,
        )
        self.evolution = EvolutionStrategy(
            EvolutionConfig(
                population_size=config.population_size,
                elite_count=config.elite_count,
                elite_preservation=config.elite_preservation,
                selection_strategy=config.selection_strategy,
                tournament_size=config.tournament_size,
                rank_weight=config.rank_weight,
                crossover_strategy=config.crossover_strategy,
                crossover_rate=config.crossover_rate,
                blend_alpha=config.blend_alpha,
                mutation_strategy=config.mutation_strategy,
                mutation_rate=config.mutation_rate,
                mutation_std=config.mutation_std,
                min_mutation_std=config.min_mutation_std,
                max_mutation_std=config.max_mutation_std,
                adaptive_mutation=config.adaptive_mutation,
            )
        )
        self.runner = ParallelRunner(
            num_workers=config.num_workers,
            timeout=config.timeout,
        )
        self.runner.start()

        # Training state
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_agent: Optional[AgentRecord] = None
        self.patience_counter = 0
        self.running = False

        # Output directory
        self.runs_dir = Path(config.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging for the trainer."""
        log_file = self.runs_dir / "training.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)

    def train(self) -> None:
        """Run the full evolution training loop.

        This is the main entry point for training. It iterates through
        generations, evaluating fitness and evolving the population.
        """
        self.running = True

        # Initialize or resume
        if self.config.resume_from and os.path.exists(self.config.resume_from):
            self._resume_from_checkpoint()
            logger.info(f"Resumed from checkpoint at generation {self.generation}")
        else:
            self.population.initialize(seed=self.config.seed)
            logger.info("Initialized new population")

        start_time = time.time()

        for gen in range(self.generation, self.config.max_generations):
            if not self.running:
                logger.info("Training stopped by user")
                break

            gen_start = time.time()
            logger.info(f"=== Generation {gen + 1} / {self.config.max_generations} ===")

            # 1. Evaluate fitness
            weights = self.population.get_population_weights()
            results = self._evaluate_population(weights)

            # 2. Update population
            fitnesses = [r.fitness for r in results]
            self.population.evaluate(fitnesses)

            # 3. Track best agent
            current_best = self.population.get_best()
            if current_best is not None and current_best.fitness > self.best_fitness:
                improvement = current_best.fitness - self.best_fitness
                if improvement >= self.config.early_stopping_min_improvement:
                    self.patience_counter = 0
                    self.best_fitness = current_best.fitness
                    self.best_agent = current_best
                    self._save_best_agent()
                    logger.info(f"New best fitness: {self.best_fitness:.3f} "
                              f"(improvement: {improvement:.3f})")
                else:
                    self.patience_counter += 1
            else:
                self.patience_counter += 1

            # 4. Log metrics
            if (gen + 1) % self.config.log_interval == 0:
                self._log_generation(gen, gen_start)

            # 5. Check early stopping
            if self._check_early_stopping():
                logger.info(f"Early stopping triggered at generation {gen + 1}")
                break

            # 6. Checkpoint
            if (gen + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(gen)

            # 7. Evolve next generation
            population_weights = self.population.get_population_weights()
            new_weights = self.evolution.evolve(
                population_weights, fitnesses,
                current_fitness=self.best_fitness,
            )
            self.population.set_population_weights(new_weights)
            self.population.generation = gen + 1

            elapsed = time.time() - gen_start
            logger.info(f"Generation {gen + 1} completed in {elapsed:.1f}s")

        # Final summary
        elapsed = time.time() - start_time
        logger.info(f"=== Training complete ===")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Final generation: {self.generation}")
        logger.info(f"Best fitness: {self.best_fitness:.3f}")

        if self.best_agent is not None:
            logger.info(f"Best agent ID: {self.best_agent.agent_id}")

        self.runner.shutdown()

    def stop(self) -> None:
        """Stop the training loop."""
        self.running = False

    def _evaluate_population(self, weights: List[np.ndarray]) -> List[MatchResult]:
        """Evaluate fitness of the population.

        Args:
            weights: List of weight arrays.

        Returns:
            List of MatchResult for each agent.
        """
        # Determine match duration
        duration_map = {
            "full": 1800,
            "short": 600,
            "overtime": 2400,
        }
        duration = duration_map.get(self.config.match_duration, 1800)

        # Evaluate in batches if needed
        results = []
        batch_size = min(self.config.batch_size, len(weights))

        for i in range(0, len(weights), batch_size):
            batch = weights[i:i + batch_size]
            batch_results = self.runner.evaluate_population(
                population_weights=batch,
                matches_per_agent=self.config.matches_per_agent,
                opponent_type="random",
                seed=self.config.seed + i,
            )
            results.extend(batch_results)

        return results

    def _check_early_stopping(self) -> bool:
        """Check if early stopping should be triggered."""
        return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, gen: int) -> None:
        """Save a training checkpoint.

        Args:
            gen: Current generation number.
        """
        gen_dir = self.runs_dir / f"gen_{gen + 1:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save population
        pop_path = gen_dir / "population.pt"
        self.population.save_checkpoint(str(pop_path))

        # Save config
        config_path = gen_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save fitness history
        history_path = gen_dir / "fitness_history.json"
        with open(history_path, "w") as f:
            json.dump(self.population.fitness_history, f, indent=2)

        # Save metrics
        stats = self.population.get_statistics()
        metrics_path = gen_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Clean old checkpoints
        self._cleanup_checkpoints()

        logger.info(f"Saved checkpoint for generation {gen + 1}")

    def _save_best_agent(self) -> None:
        """Save the best agent to the 'best' directory."""
        best_dir = self.runs_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        if self.best_agent is None:
            return

        # Save agent weights
        agent_path = best_dir / "best_agent.pt"
        self.best_agent.agent.save_checkpoint(str(agent_path))

        # Save metadata
        meta_path = best_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "generation": self.generation,
                "fitness": self.best_fitness,
                "agent_id": self.best_agent.agent_id,
            }, f, indent=2)

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        gen_dirs = sorted(self.runs_dir.glob("gen_*"))
        if len(gen_dirs) > self.config.max_checkpoints:
            for old_dir in gen_dirs[:len(gen_dirs) - self.config.max_checkpoints]:
                import shutil
                shutil.rmtree(old_dir)
                logger.info(f"Removed old checkpoint: {old_dir}")

    def _log_generation(self, gen: int, gen_start: float) -> None:
        """Log generation metrics.

        Args:
            gen: Current generation number.
            gen_start: Start time of the generation.
        """
        stats = self.population.get_statistics()
        elapsed = time.time() - gen_start

        logger.info(
            f"Gen {gen + 1} | Best: {stats.get('best_fitness', 0):.3f} | "
            f"Mean: {stats.get('mean_fitness', 0):.3f} | "
            f"Median: {stats.get('median_fitness', 0):.3f} | "
            f"Min: {stats.get('min_fitness', 0):.3f} | "
            f"Std: {stats.get('std_fitness', 0):.3f} | "
            f"Diversity: {stats.get('diversity', 0):.3f} | "
            f"Time: {elapsed:.1f}s"
        )

    def _resume_from_checkpoint(self) -> None:
        """Resume training from a saved checkpoint."""
        if not self.config.resume_from:
            return

        # Load population
        self.population.load_checkpoint(self.config.resume_from)

        # Find the best generation directory for config
        gen_dirs = sorted(self.runs_dir.glob("gen_*"), reverse=True)
        if gen_dirs:
            config_path = gen_dirs[0] / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    saved_config = json.load(f)
                self.config = TrainingConfig.from_dict(saved_config)

        self.generation = self.population.generation

        # Find best agent
        best_dir = self.runs_dir / "best"
        if best_dir.exists():
            meta_path = best_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.best_fitness = meta["fitness"]

    def get_training_status(self) -> Dict:
        """Get current training status.

        Returns:
            Dictionary of training status.
        """
        stats = self.population.get_statistics()
        return {
            "generation": self.generation,
            "max_generations": self.config.max_generations,
            "best_fitness": self.best_fitness,
            "current_best": stats.get("best_fitness", 0),
            "mean_fitness": stats.get("mean_fitness", 0),
            "patience_counter": self.patience_counter,
            "early_stopping_patience": self.config.early_stopping_patience,
            "running": self.running,
            "population_size": len(self.population),
            "diversity": stats.get("diversity", 0),
            "elapsed": time.time() - (getattr(self, '_start_time', time.time())),
        }

    def __del__(self) -> None:
        """Clean up resources."""
        self.runner.shutdown()
