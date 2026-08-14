"""Main evolution trainer for CR-Pipeline.

Orchestrates the full evolutionary training loop:
1. Initialize population
2. For each generation:
   a. Evaluate fitness (parallel simulation)
   b. Update diversity and speciation
   c. Apply evolution operators
   d. Log metrics and checkpoint
3. Handle early stopping and resume

Enhanced with:
- Adaptive mutation rate tracking
- Diversity-aware evolution
- Better fitness aggregation
- Comprehensive metrics logging
- Checkpoint versioning
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from ..models import (
    Population, AgentRecord, EvolutionStrategy, EvolutionConfig,
)
from ..env.sim import ParallelRunner, MatchResult, CARD_DEFS
from ..alerting import AlertManager, AlertRule, AlertType, AlertLevel
from ..registry import ModelRegistry, ModelStage
from .monitoring import MetricsCollector, BottleneckDetector, MonitoringConfig
from .data_pipeline import MatchCollector

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
        diversity_preservation: Whether to maintain diversity.
        diversity_threshold: Threshold for speciation.
        novelty_search: Whether to use novelty search.
        opponent_type: Default opponent type for evaluation.
        curriculum_learning: Whether to use curriculum learning.
        phase_transitions: Dict of phase transition points.
        adaptive_population: Whether to adapt population size.
        min_population_size: Minimum population size.
        max_population_size: Maximum population size.
        population_growth_rate: Rate of population growth per phase.
        use_training_decks: Whether to use varied training decks for opponents.
        tournament_mode: Train by agent-vs-agent tournament play (default).
            When False, fitness comes from matches against scripted opponents.
        tournament_format: swiss, round_robin, single_elim, double_elim, league.
        tournament_matches: Matches per pairing, split between sides.
        tournament_elite_fraction: Fraction of population preserved as elites.
        tournament_rounds: Swiss rounds; None picks ceil(log2(entrants)).
        hall_of_fame_size: Past champions entered into each tournament as
            benchmark opponents. Keeps ratings anchored across generations and
            stops the population from cycling against its own current meta.
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
    diversity_preservation: bool = False
    diversity_threshold: float = 0.5
    novelty_search: bool = False
    opponent_type: str = "random"
    curriculum_learning: bool = False
    phase_transitions: Dict[str, int] = field(default_factory=lambda: {
        "phase1_baseline": 10,
        "phase2_basic": 60,
        "phase3_advanced": 160,
        "phase4_finetune": 260,
        "phase5_competitive": 360,
    })
    adaptive_population: bool = False
    min_population_size: int = 50
    max_population_size: int = 300
    population_growth_rate: float = 0.5
    use_training_decks: bool = True
    # Tournament mode
    # Agents are trained by playing each other. Fitness is a competitor's
    # standing in a Swiss tournament against the rest of the population plus
    # the hall of fame, not a score against scripted bots.
    tournament_mode: bool = True
    tournament_format: str = "swiss"
    tournament_matches: int = 2
    tournament_elite_fraction: float = 0.1
    tournament_rounds: Optional[int] = None   # None -> ceil(log2(entrants))
    hall_of_fame_size: int = 4
    # Monitoring
    monitor_resources: bool = False
    monitor_sample_interval: float = 1.0
    monitor_output_dir: str = "runs/monitoring"
    # Alerting
    enable_alerts: bool = True
    alert_log_path: str = "runs/alerts.log"
    # Registry
    enable_registry: bool = False
    registry_dir: str = "runs/model_registry"
    # Data collection
    collect_matches: bool = False
    match_output_dir: str = "runs/matches"

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
            "resume_from": self.resume_from,
            "seed": self.seed,
            "diversity_preservation": self.diversity_preservation,
            "diversity_threshold": self.diversity_threshold,
            "novelty_search": self.novelty_search,
            "opponent_type": self.opponent_type,
            "curriculum_learning": self.curriculum_learning,
            "phase_transitions": self.phase_transitions,
            "adaptive_population": self.adaptive_population,
            "min_population_size": self.min_population_size,
            "max_population_size": self.max_population_size,
            "population_growth_rate": self.population_growth_rate,
            "use_training_decks": self.use_training_decks,
            # Tournament mode
            "tournament_mode": self.tournament_mode,
            "tournament_format": self.tournament_format,
            "tournament_matches": self.tournament_matches,
            "tournament_elite_fraction": self.tournament_elite_fraction,
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
    - Evolution operations with diversity tracking
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
                diversity_preservation=config.diversity_preservation,
                diversity_threshold=config.diversity_threshold,
                novelty_search=config.novelty_search,
                seed=config.seed,
            )
        )

        # Tournament mode support
        self.tournament_mode = config.tournament_mode
        self.tournament_evaluator = None
        self.tournament_strategy = None
        if self.tournament_mode:
            from ..train import FitnessEvaluator
            from ..models import TournamentEvolutionStrategy
            self.tournament_evaluator = FitnessEvaluator(
                num_workers=config.num_workers,
                matches_per_agent=config.tournament_matches,
            )
            self.tournament_evaluator.runner.start()
            self.tournament_strategy = TournamentEvolutionStrategy(
                tournament_format=config.tournament_format,
                matches_per_pair=config.tournament_matches,
                elite_fraction=config.tournament_elite_fraction,
                seed=config.seed,
            )
            logger.info("Tournament mode enabled")

        self.runner = ParallelRunner(
            num_workers=config.num_workers,
            timeout=config.timeout,
        )
        self.runner.start()

        # Training state
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_agent: Optional[AgentRecord] = None
        # Copy of the best genome seen, independent of population mutation.
        self.best_genome: Optional[np.ndarray] = None
        self.best_agent_id: Optional[str] = None
        self.best_generation: int = 0
        self.patience_counter = 0

        # Tournament state. The hall of fame holds (genome, metadata) for past
        # champions, which enter each tournament as benchmark opponents.
        self.hall_of_fame: List[Tuple[np.ndarray, dict]] = []
        self.elo_ratings: Dict[str, float] = {}
        self.elo_history: Dict[str, List[float]] = {}
        self.last_tournament = None

        # Optional observer called once per generation with a progress dict.
        # Set it directly on the instance; see _emit_progress for the payload.
        self.on_generation: Optional[Callable[[dict], None]] = None
        self.running = False
        self._start_time: float = 0.0

        # Output directory
        self.runs_dir = Path(config.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring
        self.monitoring = None
        self.bottleneck_detector = None
        if config.monitor_resources:
            self.monitoring = MetricsCollector(MonitoringConfig(
                enabled=True,
                sample_interval_sec=config.monitor_sample_interval,
                output_dir=config.monitor_output_dir,
            ))
            self.monitoring.start()
            self.bottleneck_detector = BottleneckDetector()
            logger.info("Resource monitoring enabled")

        # Initialize alerting
        self.alert_manager = None
        if config.enable_alerts:
            self.alert_manager = AlertManager()
            for rule in self.alert_manager.get_default_rules():
                self.alert_manager.add_rule(rule)
            logger.info("Alerting enabled with default rules")

        # Initialize model registry
        self.registry = None
        if config.enable_registry:
            self.registry = ModelRegistry(config.registry_dir)
            logger.info("Model registry enabled")

        # Initialize match collector
        self.match_collector = None
        if config.collect_matches:
            self.match_collector = MatchCollector()
            logger.info("Match collection enabled")

        # Logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Attach a per-run file handler to the module logger."""
        log_file = self.runs_dir / "training.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)
        # Tracked so close() can detach it. The handler is added to a
        # module-level logger, so constructing several trainers in one process
        # otherwise stacks handlers (duplicating every line) and leaks an open
        # file per run -- which on Windows also blocks removing the run dir.
        self._log_handler = handler

    def close(self) -> None:
        """Release the run's log handler and shut down worker processes."""
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
            self._log_handler = None

        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.shutdown()

        evaluator = getattr(self, "tournament_evaluator", None)
        if evaluator is not None:
            evaluator.shutdown()

    def __enter__(self) -> "EvolutionTrainer":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def train(self) -> None:
        """Run the full evolution training loop.

        This is the main entry point for training. It iterates through
        generations, evaluating fitness and evolving the population.
        """
        self.running = True
        self._start_time = time.time()

        # Initialize or resume
        if self.config.resume_from and os.path.exists(self.config.resume_from):
            self._resume_from_checkpoint()
            logger.info(f"Resumed from checkpoint at generation {self.generation}")
        else:
            self.population.initialize(seed=self.config.seed)
            logger.info("Initialized new population")

        for gen in range(self.generation, self.config.max_generations):
            if not self.running:
                logger.info("Training stopped by user")
                break

            gen_start = time.time()
            logger.info(f"=== Generation {gen + 1} / {self.config.max_generations} ===")

            # 1. Evaluate fitness
            weights = self.population.get_population_weights()
            results = self._evaluate_population(weights, generation=gen)

            # 2. Update population with fitness
            fitnesses = [r.fitness for r in results]
            self.population.evaluate(fitnesses)

            # 3. Update diversity and speciation
            if self.config.diversity_preservation:
                diversity = self.population.update_diversity()
                species = self.population.speciate(self.config.diversity_threshold)
                logger.info(f"Diversity: {diversity:.4f}, Species: {len(species)}")

            # 4. Track best agent.
            # In tournament mode fitness is a *standing within this
            # generation's field*, so it is not comparable across generations:
            # the top agent of a weak generation can out-score the top agent of
            # a strong one. ELO is the cross-generation measure, anchored by the
            # hall of fame playing in every tournament, so rank by that instead.
            current_best = self.population.get_best()
            current_score = self._comparable_score(current_best, results)
            if current_best is not None and current_score > self.best_fitness:
                improvement = current_score - self.best_fitness
                if improvement >= self.config.early_stopping_min_improvement:
                    self.patience_counter = 0
                    self.best_fitness = current_score
                    self.best_agent = current_best
                    # Snapshot the genome. AgentRecords are reused in place by
                    # set_population_weights, so holding only the reference
                    # meant `best_agent.genome` was silently overwritten by a
                    # later generation and no longer described the best agent.
                    self.best_genome = (
                        None if current_best.genome is None
                        else np.array(current_best.genome, copy=True)
                    )
                    self.best_agent_id = current_best.agent_id
                    self.best_generation = gen + 1
                    current_best.agent.update_best(current_best.fitness)
                    self._save_best_agent()
                    logger.info(f"New best fitness: {self.best_fitness:.3f} "
                              f"(improvement: {improvement:.3f})")
                else:
                    self.patience_counter += 1
            else:
                self.patience_counter += 1

            # 4b. Report progress to any observer (the desktop UI uses this),
            # and refresh the run-level summary so tools can read a run while
            # it is still going.
            self._emit_progress(gen, fitnesses, results)
            self._save_run_summary(gen)

            # 5. Check for phase transition
            if self.config.curriculum_learning:
                self._check_phase_transition(gen)

            # 5b. Record monitoring data
            if self.monitoring:
                diversity = float(np.std(fitnesses)) if fitnesses else 0.0
                self.monitoring.record_generation(gen + 1, fitnesses, diversity)
                # Check bottlenecks
                gen_time = time.time() - gen_start
                if self.bottleneck_detector:
                    bottlenecks = self.bottleneck_detector.detect(
                        gpu_compute_pct=0, gpu_memory_pct=0,
                        cpu_percent=0, gen_time_sec=gen_time,
                    )
                    # Check for alerts from bottlenecks
                    if self.alert_manager:
                        for bn in bottlenecks:
                            if bn.severity in ("high", "critical"):
                                self.alert_manager.check({
                                    "generation": gen + 1,
                                    "bottleneck_type": bn.bottleneck_type.name,
                                    "bottleneck_severity": bn.severity,
                                    "message": bn.description,
                                })

            # 5c. Register best model in registry
            if self.registry and current_best is not None:
                model_id = f"gen_{gen + 1:04d}"
                self.registry.register_model(
                    model_id=model_id,
                    version=f"v{gen + 1}",
                    fitness=current_best.fitness,
                    architecture="cnn_lstm",
                    tags=[f"generation_{gen + 1}"],
                )

            # 5d. Collect match data if enabled
            if self.match_collector and results:
                for result in results:
                    if hasattr(result, 'state_snapshots') and result.state_snapshots:
                        self.match_collector.collect_match(
                            generation=gen + 1,
                            agent1_id=current_best.agent_id if current_best else "unknown",
                            agent2_id=result.opponent_id if hasattr(result, 'opponent_id') else "opponent",
                            state_snapshots=result.state_snapshots[-50:] if result.state_snapshots else [],
                            action_history=result.action_history[-200:] if hasattr(result, 'action_history') and result.action_history else [],
                            reward_history=result.reward_history[-1000:] if hasattr(result, 'reward_history') and result.reward_history else [],
                            winner=result.winner if hasattr(result, 'winner') else None,
                            agent1_trophies=result.agent1_trophies if hasattr(result, 'agent1_trophies') else 0,
                            agent2_trophies=result.agent2_trophies if hasattr(result, 'agent2_trophies') else 0,
                            duration_ticks=result.duration_ticks if hasattr(result, 'duration_ticks') else 0,
                        )

            # 6. Log metrics
            if (gen + 1) % self.config.log_interval == 0:
                self._log_generation(gen, gen_start)

            # 7. Check early stopping
            if self._check_early_stopping():
                logger.info(f"Early stopping triggered at generation {gen + 1}")
                break

            # 8. Checkpoint
            if (gen + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(gen)

            # 9. Evolve next generation
            population_weights = self.population.get_population_weights()
            avg_fitness = float(np.mean(fitnesses))

            if self.tournament_mode and self.tournament_strategy is not None:
                # Fitness already *is* the tournament standing, so selection on
                # it is tournament-driven evolution. Handing the population to
                # TournamentEvolutionStrategy here would run a second full
                # tournament and pay for the matchmaking twice.
                new_weights, evolution_info = self.tournament_strategy.evolve(
                    population=population_weights,
                    weights_list=population_weights,
                    current_fitnesses=fitnesses,
                    evaluator=None,
                    generation=gen,
                )
                if self.last_tournament is not None:
                    ratings = self.last_tournament.elo_ratings
                    logger.info(
                        "Tournament: %d matches, elo %.0f-%.0f, champion %s",
                        self.last_tournament.total_matches,
                        min(ratings.values()), max(ratings.values()),
                        self.last_tournament.rankings[0][0],
                    )
            else:
                new_weights, evolution_info = self.evolution.evolve(
                    population_weights, fitnesses,
                    current_fitness=self.best_fitness,
                    avg_fitness=avg_fitness,
                )
            self.population.set_population_weights(new_weights)
            self.population.generation = gen + 1

            elapsed = time.time() - gen_start
            logger.info(f"Generation {gen + 1} completed in {elapsed:.1f}s")

        # Final summary
        elapsed = time.time() - self._start_time
        logger.info(f"=== Training complete ===")
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"Final generation: {self.generation}")
        logger.info(f"Best fitness: {self.best_fitness:.3f}")

        if self.best_agent is not None:
            logger.info(f"Best agent ID: {self.best_agent.agent_id}")

        # Shutdown monitoring
        if self.monitoring:
            self.monitoring.stop()
            logger.info("Monitoring stopped")

        # Save alert history
        if self.alert_manager:
            self.alert_manager.save_history(str(self.runs_dir / "alert_history.json"))
            alerts = self.alert_manager.check({
                "training_complete": True,
                "best_fitness": self.best_fitness,
                "total_gens": self.generation,
                "elapsed_min": elapsed / 60,
            })
            logger.info(f"Alert history saved ({len(alerts)} final alerts)")

        # Shutdown tournament evaluator if active
        if self.tournament_evaluator is not None:
            self.tournament_evaluator.shutdown()

        self.runner.shutdown()

    def stop(self) -> None:
        """Stop the training loop."""
        self.running = False

    def _evaluate_population(self, weights: List[np.ndarray],
                             generation: int = 0) -> List[MatchResult]:
        """Score the population for this generation.

        In tournament mode (the default) agents are ranked by playing each
        other; otherwise they are scored against scripted opponents.

        Args:
            weights: List of weight arrays.
            generation: Current generation, used to advance the shared seed.

        Returns:
            List of MatchResult for each agent, index-aligned with ``weights``.
        """
        if self.tournament_mode and self.tournament_evaluator is not None:
            return self._evaluate_by_tournament(weights, generation)
        return self._evaluate_against_scripted(weights, generation)

    def _evaluate_by_tournament(self, weights: List[np.ndarray],
                                generation: int) -> List[MatchResult]:
        """Rank the population by having its members play each other.

        Entrants are the whole population plus the hall of fame -- previous
        generations' champions, which do not reproduce but do provide a fixed
        reference to be measured against. Without them, fitness is purely
        relative to the current population and a whole generation can drift
        without getting stronger, or cycle against its own meta.

        Fitness is the agent's tournament standing (points per match plus a
        small tower-differential term), so selection acts directly on the
        thing being optimised: winning games against real opponents.
        """
        population_size = len(weights)
        entrant_weights = list(weights) + [g for g, _ in self.hall_of_fame]
        # Hall-of-fame entrants keep the id they were admitted under, so their
        # carried-over rating follows the genome. Numbering them by slot would
        # hand a departing champion's rating to whoever rotated into its place.
        entrant_ids = (
            [f"agent_{i}" for i in range(population_size)]
            + [meta["id"] for _, meta in self.hall_of_fame]
        )

        result = self.tournament_evaluator.run_tournament(
            agent_ids=entrant_ids,
            weights_list=entrant_weights,
            matches_per_pair=self.config.tournament_matches,
            seed=self.config.seed + generation * 1000,
            format=self._tournament_format_enum(),
            rounds=self.config.tournament_rounds,
            generation=generation,
            initial_elo=self.elo_ratings,
        )

        self.last_tournament = result
        # Ratings persist across generations for the hall of fame, which is
        # what makes ELO comparable over a run rather than per-generation noise.
        self.elo_ratings.update(result.elo_ratings)
        for agent_id, rating in result.elo_ratings.items():
            self.elo_history.setdefault(agent_id, []).append(rating)

        results = []
        for i in range(population_size):
            agent_id = f"agent_{i}"
            stats = result.agent_stats[agent_id]
            results.append(MatchResult(
                agent_id=agent_id,
                fitness=stats.compute_composite_score(),
                wins=stats.wins,
                draws=stats.draws,
                losses=stats.losses,
                avg_towers_destroyed=stats.towers_destroyed / max(1, stats.matches),
                avg_duration=stats.total_duration / max(1, stats.matches),
                metadata={"elo": result.elo_ratings.get(agent_id, 1500.0),
                          "rank": result.get_ranking(agent_id)},
            ))

        self._update_hall_of_fame(weights, results, generation)
        return results

    def _save_run_summary(self, gen: int) -> None:
        """Write run-level ``fitness_history.json``/``metrics.json``.

        Previously these existed only inside ``gen_XXXX/`` checkpoint folders,
        so a run directory carried no top-level record of itself -- tooling
        that reads a run (the dashboard, the desktop UI) found nothing, and a
        run with checkpointing turned off left no history at all.
        """
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.runs_dir / "fitness_history.json", "w") as handle:
                json.dump(self.population.fitness_history, handle, indent=2)

            metrics = {
                "generation": gen + 1,
                "max_generations": self.config.max_generations,
                "best_score": self.best_fitness,
                "best_score_kind": "elo" if self.tournament_mode else "fitness",
                "best_generation": self.best_generation,
                "tournament_mode": self.tournament_mode,
                "tournament_format": self.config.tournament_format,
                "population_size": self.config.population_size,
                "elapsed_seconds": time.time() - self._start_time,
            }
            with open(self.runs_dir / "metrics.json", "w") as handle:
                json.dump(metrics, handle, indent=2)

            if self.elo_history:
                with open(self.runs_dir / "elo_history.json", "w") as handle:
                    json.dump(self.elo_history, handle, indent=2)
        except OSError:
            logger.warning("Could not write run summary to %s", self.runs_dir)

    def _emit_progress(self, gen: int, fitnesses: List[float],
                       results: List[MatchResult]) -> None:
        """Hand a snapshot of this generation to ``on_generation``, if set.

        Lets a caller (the desktop UI) follow a run without reaching into
        trainer internals or scraping the log. Observer failures are logged
        and swallowed: a broken progress display must not abort training.
        """
        callback = getattr(self, "on_generation", None)
        if callback is None:
            return

        champion = max(results, key=lambda r: r.fitness) if results else None
        snapshot = {
            "generation": gen + 1,
            "total_generations": self.config.max_generations,
            "best_fitness": float(max(fitnesses)) if fitnesses else 0.0,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "min_fitness": float(min(fitnesses)) if fitnesses else 0.0,
            "std_fitness": float(np.std(fitnesses)) if fitnesses else 0.0,
            "champion_id": champion.agent_id if champion else None,
            "champion_elo": (champion.metadata.get("elo") if champion else None),
            "champion_record": (
                f"{champion.wins}W/{champion.draws}D/{champion.losses}L"
                if champion else None
            ),
            "best_overall": float(self.best_fitness),
            "best_generation": self.best_generation,
            "tournament_matches": (
                self.last_tournament.total_matches if self.last_tournament else 0
            ),
            "hall_of_fame": len(self.hall_of_fame),
            "elapsed": time.time() - self._start_time,
        }

        try:
            callback(snapshot)
        except Exception:  # pragma: no cover - defensive
            logger.exception("on_generation callback failed; continuing training")

    def _comparable_score(self, record, results: List[MatchResult]) -> float:
        """A score for ``record`` that is comparable across generations.

        Tournament fitness is a standing relative to this generation's field,
        so it says nothing about whether generation 9 is stronger than
        generation 2. ELO is comparable because the hall of fame enters every
        tournament, giving successive populations a shared reference to be
        rated against. Outside tournament mode, fitness is already absolute
        (a record against fixed scripted opponents).
        """
        if record is None:
            return -float("inf")
        if not (self.tournament_mode and self.tournament_evaluator is not None):
            return record.fitness

        # results are index-aligned with population.agents (ranking sorts a
        # copy, so the population order is stable).
        for index, candidate in enumerate(self.population.agents):
            if candidate is record and index < len(results):
                return float(results[index].metadata.get("elo", record.fitness))
        return record.fitness

    def _tournament_format_enum(self):
        """Map the configured format name onto a TournamentFormat."""
        from ..train.evaluator import TournamentFormat

        name = str(self.config.tournament_format).lower()
        mapping = {
            "swiss": TournamentFormat.SWISS,
            "round_robin": TournamentFormat.ROUND_ROBIN,
            "single_elim": TournamentFormat.SINGLE_ELIMINATION,
            "double_elim": TournamentFormat.DOUBLE_ELIMINATION,
            "league": TournamentFormat.LEAGUE,
        }
        if name not in mapping:
            raise ValueError(
                f"Unknown tournament_format {self.config.tournament_format!r}; "
                f"expected one of {sorted(mapping)}"
            )
        return mapping[name]

    def _update_hall_of_fame(self, weights: List[np.ndarray],
                             results: List[MatchResult],
                             generation: int) -> None:
        """Admit this generation's champion to the hall of fame."""
        if self.config.hall_of_fame_size <= 0 or not results:
            return

        champion_idx = max(range(len(results)), key=lambda i: results[i].fitness)
        self.hall_of_fame.append((
            np.array(weights[champion_idx], copy=True),
            {"id": f"hof_gen{generation}",
             "generation": generation,
             "fitness": results[champion_idx].fitness,
             "elo": results[champion_idx].metadata.get("elo", 1500.0)},
        ))
        # Keep the most recent champions: older ones stop being informative
        # opponents once the population has moved past them.
        if len(self.hall_of_fame) > self.config.hall_of_fame_size:
            self.hall_of_fame = self.hall_of_fame[-self.config.hall_of_fame_size:]

    def _evaluate_against_scripted(self, weights: List[np.ndarray],
                                   generation: int = 0) -> List[MatchResult]:
        """Score every agent against scripted opponents.

        All agents in a generation are scored on the same match seeds (common
        random numbers), so their fitness is directly comparable. The seed
        advances per generation, so the population is not repeatedly graded on
        one fixed set of matches.
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

        # Shared across the whole population, advanced per generation. Batching
        # must not change it, or agents in different batches would face
        # different matches and stop being comparable.
        generation_seed = self.config.seed + generation * 1000

        for i in range(0, len(weights), batch_size):
            batch = weights[i:i + batch_size]
            batch_results = self.runner.evaluate_population(
                population_weights=batch,
                matches_per_agent=self.config.matches_per_agent,
                opponent_type=self.config.opponent_type,
                seed=generation_seed,
            )
            results.extend(batch_results)

        return results

    def _check_early_stopping(self) -> bool:
        """Check if early stopping should be triggered."""
        return self.patience_counter >= self.config.early_stopping_patience

    def _check_phase_transition(self, gen: int) -> str:
        """Check if we should transition to a new training phase.

        Monitors convergence and triggers phase transitions when:
        - Current phase has been completed for enough generations
        - Fitness has stabilized (converged) in current phase

        Args:
            gen: Current generation number.

        Returns:
            Current phase name.
        """
        current_phase = "phase1_baseline"
        
        for phase_name, threshold in self.config.phase_transitions.items():
            if gen >= threshold:
                current_phase = phase_name

        # Check if we should advance to next phase
        phases = list(self.config.phase_transitions.keys())
        current_idx = phases.index(current_phase) if current_phase in phases else -1
        
        if current_idx < len(phases) - 1:
            next_phase = phases[current_idx + 1]
            next_threshold = self.config.phase_transitions[next_phase]
            
            # Check if we've been in current phase long enough
            phase_start = self.config.phase_transitions[current_phase] if current_phase in self.config.phase_transitions else 0
            phase_duration = gen - phase_start
            
            # Check convergence in current phase
            if len(self.population.fitness_history.get("best", [])) >= 10:
                recent_best = self.population.fitness_history["best"][-10:]
                recent_change = max(recent_best) - min(recent_best)
                
                # If converged (low variance) and phase duration sufficient
                if recent_change < 0.5 and phase_duration >= 5:
                    logger.info(f"Phase transition: {current_phase} -> {next_phase} "
                              f"(converged, change={recent_change:.3f})")
                    self._apply_phase_config(next_phase)
                    return next_phase
        
        return current_phase

    def _apply_phase_config(self, phase_name: str) -> None:
        """Apply configuration changes for a new training phase."""
        if phase_name not in PHASE_CONFIGS:
            return
        
        phase_config = PHASE_CONFIGS[phase_name]
        
        # Update evolution parameters
        if "evolution" in phase_config:
            evol = phase_config["evolution"]
            if "population_size" in evol:
                self.config.population_size = evol["population_size"]
            if "elite_count" in evol:
                self.config.elite_count = evol["elite_count"]
            if "mutation_rate" in evol:
                self.config.mutation_rate = evol["mutation_rate"]
            if "mutation_std" in evol:
                self.config.mutation_std = evol["mutation_std"]
            if "adaptive_mutation" in evol:
                self.config.adaptive_mutation = evol["adaptive_mutation"]
            if "crossover_rate" in evol:
                self.config.crossover_rate = evol["crossover_rate"]
            if "max_generations" in evol:
                self.config.max_generations = max(self.config.max_generations, evol["max_generations"])
            if "matches_per_agent" in evol:
                self.config.matches_per_agent = evol["matches_per_agent"]
            if "num_workers" in evol:
                self.config.num_workers = evol["num_workers"]
        
        # Update simulation parameters
        if "sim" in phase_config:
            sim = phase_config["sim"]
            if "match_duration" in sim:
                self.config.match_duration = sim["match_duration"]
        
        logger.info(f"Applied phase config: {phase_name}")
        logger.info(f"  Population: {self.config.population_size}")
        logger.info(f"  Elite count: {self.config.elite_count}")
        logger.info(f"  Mutation rate: {self.config.mutation_rate}")
        logger.info(f"  Mutation std: {self.config.mutation_std}")
        logger.info(f"  Matches per agent: {self.config.matches_per_agent}")

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

        # Save diversity history
        diversity_path = gen_dir / "diversity_history.json"
        with open(diversity_path, "w") as f:
            json.dump({"diversity": self.population.diversity_history}, f, indent=2)

        # Save metrics
        stats = self.population.get_statistics()
        metrics_path = gen_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(stats, f, indent=2)

        # Save evolution config summary
        Evo_summary = self.evolution.get_config_summary()
        with open(gen_dir / "evolution_config.json", "w") as f:
            json.dump(Evo_summary, f, indent=2)

        # Clean old checkpoints
        self._cleanup_checkpoints()

        logger.info(f"Saved checkpoint for generation {gen + 1}")

    def _save_best_agent(self) -> None:
        """Save the best agent to the 'best' directory."""
        best_dir = self.runs_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        if self.best_agent is None:
            return

        # The snapshotted genome is authoritative: it is what was evaluated and
        # ranked. Push it onto the agent before saving so the checkpoint holds
        # the trained policy rather than an untouched Torch network.
        if self.best_genome is not None:
            self.best_agent.agent.weights = np.array(self.best_genome, copy=True)

        agent_path = best_dir / "best_agent.pt"
        self.best_agent.agent.save_checkpoint(str(agent_path))

        # Save metadata
        meta_path = best_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "generation": self.generation,
                "best_generation": self.best_generation,
                "policy_params": (
                    int(self.best_genome.size)
                    if self.best_genome is not None else None
                ),
                "fitness": self.best_fitness,
                "agent_id": self.best_agent.agent_id,
                "timestamp": time.time(),
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
            f"Species: {stats.get('num_species', 0)} | "
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
            "num_species": stats.get("num_species", 0),
            "elapsed": time.time() - self._start_time,
        }

    def __del__(self) -> None:
        """Clean up resources."""
        self.runner.shutdown()
