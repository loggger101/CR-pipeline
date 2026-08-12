"""Parallel simulation runner for training.

Manages multiple independent simulation instances running concurrently,
collecting fitness scores and managing worker lifecycle.

Uses multiprocessing for true parallelism across CPU cores.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .engine import SimulationEngine, SimulationStepResult

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single simulated match.

    Attributes:
        agent_id: ID of the agent that played.
        fitness: Computed fitness score.
        wins: Number of wins.
        draws: Number of draws.
        losses: Number of losses.
        avg_towers_destroyed: Average towers destroyed per match.
        avg_duration: Average match duration in ticks.
        metadata: Additional match statistics.
    """
    agent_id: str
    fitness: float
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_towers_destroyed: float = 0.0
    avg_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerConfig:
    """Configuration for a single simulation worker.

    Attributes:
        seed: Random seed for this worker.
        match_count: Number of matches to run.
        deck: Card deck to use.
        opponent_deck: Opponent card deck.
        match_duration_ticks: Length of each match.
        overtime_ticks: Overtime duration.
        elixir_regen_rate: Elixir regeneration rate.
    """
    seed: int = 42
    match_count: int = 5
    deck: Optional[List[str]] = None
    opponent_deck: Optional[List[str]] = None
    match_duration_ticks: int = 1800
    overtime_ticks: int = 120
    elixir_regen_rate: float = 0.3


def _run_matches(
    worker_id: int,
    config: WorkerConfig,
    weights: np.ndarray,
    opponent_type: str = "random",
    opponent_weights: Optional[np.ndarray] = None,
) -> MatchResult:
    """Run multiple matches in a single worker process.

    This function runs in a separate process. It creates SimulationEngine
    instances, runs matches against the specified opponent type, and
    returns aggregated results.

    Args:
        worker_id: Process ID for logging.
        config: Worker configuration.
        weights: Neural network weights for the agent.
        opponent_type: Type of opponent ("random", "greedy", "elite").
        opponent_weights: Weights for elite opponents (if opponent_type == "elite").

    Returns:
        Aggregated MatchResult.
    """
    import sys
    sys.setrecursionlimit(10000)

    logger.info(f"Worker {worker_id}: Starting {config.match_count} matches")

    engine = SimulationEngine(
        deck=config.deck,
        opponent_deck=config.opponent_deck,
        match_duration_ticks=config.match_duration_ticks,
        overtime_ticks=config.overtime_ticks,
        elixir_regen_rate=config.elixir_regen_rate,
        seed=config.seed + worker_id,
    )

    from .engine import SimulationEngine
    from .actions import Action, ActionType

    wins = 0
    draws = 0
    losses = 0
    total_towers = 0
    total_duration = 0

    for match_idx in range(config.match_count):
        state = engine.reset()

        while not engine.terminated:
            # Get action from agent
            agent_action = _select_action(state, weights, engine, opponent_type,
                                          opponent_weights)

            # Get opponent action
            if opponent_type == "random":
                opp_action = _random_opponent_action(engine)
            elif opponent_type == "greedy":
                opp_action = _greedy_opponent_action(engine)
            else:
                opp_action = _random_opponent_action(engine)

            # Execute step
            result = engine.step(agent_action, opp_action)

            # Update state
            state = engine._get_state()

        # Record results
        if engine.terminated:
            info = engine.action_history[-1] if engine.action_history else {}
            if engine.player_trophies > engine.opponent_trophies:
                wins += 1
            elif engine.opponent_trophies > engine.player_trophies:
                losses += 1
            else:
                draws += 1

            total_towers += engine.opponent_towers_destroyed
            total_duration += engine.tick

    avg_towers = total_towers / config.match_count if config.match_count > 0 else 0
    avg_dur = total_duration / config.match_count if config.match_count > 0 else 0

    # Compute fitness
    fitness = (
        wins * 1.0
        + draws * 0.5
        - losses * 0.5
        + 0.1 * avg_towers
    )

    logger.info(f"Worker {worker_id}: fitness={fitness:.3f}, "
                f"wins={wins}, draws={draws}, losses={losses}")

    return MatchResult(
        agent_id=f"worker_{worker_id}",
        fitness=fitness,
        wins=wins,
        draws=draws,
        losses=losses,
        avg_towers_destroyed=avg_towers,
        avg_duration=avg_dur,
    )


def _select_action(state, weights, engine, opponent_type, opponent_weights):
    """Select an action for the agent given current state and weights."""
    # Placeholder: will be replaced with actual inference
    # For now, return a random valid action
    return Action.pass_action()


def _random_opponent_action(engine) -> Action:
    """Generate a random valid action for the opponent."""
    import numpy as np
    rng = np.random.RandomState()

    # Random chance of playing a card
    if rng.random() < 0.3:
        # Pick a random card from hand
        card_idx = rng.randint(0, 4)
        # Pick a random valid position in opponent territory
        col = rng.randint(0, 8)
        row = rng.randint(0, 3)
        return Action.play_card(card_idx, float(col), float(row))

    return Action.pass_action()


def _greedy_opponent_action(engine) -> Action:
    """Generate a greedy action for the opponent.

    Prioritizes:
    1. Playing the most expensive affordable card
    2. Deploying near the nearest player unit
    """
    import numpy as np
    rng = np.random.RandomState()

    if rng.random() < 0.4:
        # Find most expensive affordable card
        best_card = None
        best_cost = -1
        for i, card_name in enumerate(engine.opponent_hand):
            if engine.opponent_cooldowns[i] <= 0:
                card_def = engine.opponent_deck[i] if i < len(engine.opponent_deck) else None
                if card_def and card_def.elixir_cost <= engine.opponent_elixir:
                    if card_def.elixir_cost > best_cost:
                        best_cost = card_def.elixir_cost
                        best_card = i

        if best_card is not None:
            col = rng.randint(0, 8)
            row = rng.randint(0, 3)
            return Action.play_card(best_card, float(col), float(row))

    return Action.pass_action()


class ParallelRunner:
    """Manages parallel simulation workers for fitness evaluation.

    Creates and manages a pool of worker processes, each running
    multiple matches for fitness evaluation.

    Usage:
        runner = ParallelRunner(num_workers=4)
        results = runner.run_evaluations(
            agent_weights_list=[weights1, weights2, ...],
            matches_per_agent=5,
        )
        runner.shutdown()
    """

    def __init__(self, num_workers: int = 4, timeout: int = 300):
        """Initialize the parallel runner.

        Args:
            num_workers: Number of parallel worker processes.
            timeout: Maximum seconds per evaluation batch.
        """
        self.num_workers = num_workers
        self.timeout = timeout
        self.pool: Optional[Pool] = None

    def start(self) -> None:
        """Start the worker pool."""
        self.pool = mp.Pool(processes=self.num_workers)
        logger.info(f"Started parallel runner with {self.num_workers} workers")

    def shutdown(self) -> None:
        """Shut down the worker pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            logger.info("Shut down parallel runner")

    def evaluate_population(
        self,
        population_weights: List[np.ndarray],
        matches_per_agent: int = 5,
        opponent_type: str = "random",
        opponent_weights: Optional[np.ndarray] = None,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[MatchResult]:
        """Evaluate the fitness of a population of agents.

        Args:
            population_weights: List of weight arrays for each agent.
            matches_per_agent: Number of matches per agent.
            opponent_type: Type of opponent ("random", "greedy", "elite").
            opponent_weights: Weights for elite opponents.
            deck: Card deck to use for evaluation.
            opponent_deck: Opponent card deck.
            seed: Base random seed.

        Returns:
            List of MatchResult for each agent.
        """
        if self.pool is None:
            self.start()

        results = []
        # Split work across workers
        total_matches = len(population_weights) * matches_per_agent
        matches_per_worker = max(1, total_matches // self.num_workers)

        configs = []
        worker_id = 0
        for i, weights in enumerate(population_weights):
            config = WorkerConfig(
                seed=seed + i * 1000,
                match_count=matches_per_worker,
                deck=deck,
                opponent_deck=opponent_deck,
            )
            configs.append((worker_id, config, weights))
            worker_id += 1
            if worker_id >= self.num_workers:
                worker_id = 0

        # Run evaluations
        map_args = []
        for i, (wid, cfg, wts) in enumerate(configs):
            map_args.append((wid, cfg, wts, opponent_type, opponent_weights))

        try:
            raw_results = self.pool.starmap(_run_matches, map_args,
                                            chunksize=max(1, len(map_args) // 4))
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}")
            raw_results = []

        # Aggregate results per agent
        agent_results: Dict[str, MatchResult] = {}
        for result in raw_results:
            agent_id = result.agent_id
            if agent_id not in agent_results:
                agent_results[agent_id] = MatchResult(agent_id=agent_id, fitness=0.0)
            existing = agent_results[agent_id]
            existing.wins += result.wins
            existing.draws += result.draws
            existing.losses += result.losses
            existing.avg_towers_destroyed = (
                (existing.avg_towers_destroyed + result.avg_towers_destroyed) / 2
            )
            existing.avg_duration = (existing.avg_duration + result.avg_duration) / 2
            existing.fitness = (existing.fitness + result.fitness) / 2

        results = list(agent_results.values())

        # Assign fitness scores back to agents
        for i, result in enumerate(results):
            if i < len(population_weights):
                result.agent_id = f"agent_{i}"

        return results

    def evaluate_single(
        self,
        weights: np.ndarray,
        matches: int = 5,
        opponent_type: str = "random",
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate a single agent's fitness.

        Args:
            weights: Neural network weights.
            matches: Number of matches to play.
            opponent_type: Type of opponent.
            seed: Random seed.

        Returns:
            MatchResult for the agent.
        """
        config = WorkerConfig(
            seed=seed,
            match_count=matches,
        )
        return _run_matches(0, config, weights, opponent_type)
