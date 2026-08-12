"""Fitness evaluator for trained agents.

Provides evaluation against:
- Random baseline opponents
- Greedy heuristic opponents
- Population of elite agents (self-play)
- Fixed tournament mode

Supports batch evaluation and tournament-style competitions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.sim import ParallelRunner, MatchResult, SimulationEngine
from ..env.sim.actions import Action, ActionType
from ..env.sim.state import GameStateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class TournamentResult:
    """Result of a tournament-style evaluation.

    Attributes:
        rankings: List of (agent_id, fitness) sorted by rank.
        wins: Dict of agent_id -> win count.
        draws: Dict of agent_id -> draw count.
        losses: Dict of agent_id -> loss count.
        total_matches: Total matches played.
    """
    rankings: List[Tuple[str, float]] = field(default_factory=list)
    wins: Dict[str, int] = field(default_factory=dict)
    draws: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, int] = field(default_factory=dict)
    total_matches: int = 0


class FitnessEvaluator:
    """Evaluates agent fitness against various opponents.

    Supports:
    - Single agent evaluation
    - Population evaluation
    - Tournament mode
    - Multiple opponent types
    """

    def __init__(self, num_workers: int = 4, matches_per_agent: int = 10):
        """Initialize the evaluator.

        Args:
            num_workers: Number of parallel workers.
            matches_per_agent: Default matches per agent.
        """
        self.num_workers = num_workers
        self.matches_per_agent = matches_per_agent
        self.runner = ParallelRunner(num_workers=num_workers)
        self.runner.start()

    def evaluate_against_random(
        self,
        weights: np.ndarray,
        matches: int = 10,
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent against random opponents.

        Args:
            weights: Agent weights.
            matches: Number of matches.
            seed: Random seed.

        Returns:
            MatchResult with fitness score.
        """
        result = self.runner.evaluate_single(
            weights=weights,
            matches=matches,
            opponent_type="random",
            seed=seed,
        )
        logger.info(f"Evaluated agent: fitness={result.fitness:.3f}, "
                    f"w={result.wins}, d={result.draws}, l={result.losses}")
        return result

    def evaluate_against_greedy(
        self,
        weights: np.ndarray,
        matches: int = 10,
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent against greedy opponents.

        Args:
            weights: Agent weights.
            matches: Number of matches.
            seed: Random seed.

        Returns:
            MatchResult with fitness score.
        """
        result = self.runner.evaluate_single(
            weights=weights,
            matches=matches,
            opponent_type="greedy",
            seed=seed,
        )
        logger.info(f"Evaluated agent vs greedy: fitness={result.fitness:.3f}")
        return result

    def evaluate_population(
        self,
        weights_list: List[np.ndarray],
        matches_per_agent: int = 5,
        opponent_type: str = "random",
        seed: int = 42,
    ) -> List[MatchResult]:
        """Evaluate a population of agents.

        Args:
            weights_list: List of weight arrays.
            matches_per_agent: Matches per agent.
            opponent_type: Opponent type.
            seed: Random seed.

        Returns:
            List of MatchResult for each agent.
        """
        results = self.runner.evaluate_population(
            population_weights=weights_list,
            matches_per_agent=matches_per_agent,
            opponent_type=opponent_type,
            seed=seed,
        )
        return results

    def run_tournament(
        self,
        weights_list: List[np.ndarray],
        matches_per_agent: int = 3,
        seed: int = 42,
    ) -> TournamentResult:
        """Run a tournament-style evaluation where agents play each other.

        Each agent plays against every other agent (and itself) for
        the specified number of matches.

        Args:
            weights_list: List of weight arrays.
            matches_per_agent: Matches per agent pair.
            seed: Random seed.

        Returns:
            TournamentResult with rankings and stats.
        """
        n = len(weights_list)
        wins = {f"agent_{i}": 0 for i in range(n)}
        draws = {f"agent_{i}": 0 for i in range(n)}
        losses = {f"agent_{i}": 0 for i in range(n)}
        total_matches = 0

        # Each agent plays against every other agent
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-play: play against random
                    result = self.runner.evaluate_single(
                        weights=weights_list[i],
                        matches=matches_per_agent,
                        opponent_type="random",
                        seed=seed + i * 100 + j,
                    )
                    wins[f"agent_{i}"] += result.wins
                    draws[f"agent_{i}"] += result.draws
                    losses[f"agent_{i}"] += result.losses
                    total_matches += matches_per_agent
                else:
                    # Head-to-head
                    result = self.runner.evaluate_population(
                        population_weights=[weights_list[i], weights_list[j]],
                        matches_per_agent=matches_per_agent // 2,
                        opponent_type="elite",
                        opponent_weights=weights_list[j],
                        seed=seed + i * 100 + j,
                    )
                    if result:
                        wins[f"agent_{i}"] += result[0].wins
                        draws[f"agent_{i}"] += result[0].draws
                        losses[f"agent_{i}"] += result[0].losses
                        total_matches += matches_per_agent // 2

        # Compute tournament scores
        rankings = []
        for i in range(n):
            agent_id = f"agent_{i}"
            w = wins.get(agent_id, 0)
            d = draws.get(agent_id, 0)
            l = losses.get(agent_id, 0)
            score = w * 1.0 + d * 0.5 - l * 0.5
            rankings.append((agent_id, score))

        rankings.sort(key=lambda x: x[1], reverse=True)

        return TournamentResult(
            rankings=rankings,
            wins=wins,
            draws=draws,
            losses=losses,
            total_matches=total_matches,
        )

    def compute_fitness_score(
        self,
        match_result: MatchResult,
        scoring_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute a composite fitness score from match results.

        Args:
            match_result: Match evaluation result.
            scoring_weights: Weights for each component.

        Returns:
            Composite fitness score.
        """
        if scoring_weights is None:
            scoring_weights = {
                "win_weight": 1.0,
                "draw_weight": 0.5,
                "loss_weight": -0.5,
                "tower_weight": 0.1,
            }

        score = (
            match_result.wins * scoring_weights.get("win_weight", 1.0)
            + match_result.draws * scoring_weights.get("draw_weight", 0.5)
            + match_result.losses * scoring_weights.get("loss_weight", -0.5)
            + match_result.avg_towers_destroyed * scoring_weights.get("tower_weight", 0.1)
        )

        return score

    def shutdown(self) -> None:
        """Shut down the evaluator."""
        self.runner.shutdown()

    def __del__(self) -> None:
        """Clean up resources."""
        self.runner.shutdown()
