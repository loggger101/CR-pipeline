"""Fitness evaluator for trained agents.

Provides evaluation against:
- Random baseline opponents
- Greedy heuristic opponents
- Population of elite agents (self-play)
- Tournament-style competitions with multiple formats

Enhanced with:
- Head-to-head matchups with side-swapping for fairness
- Round-robin, single/double elimination, and league tournament formats
- Detailed per-agent rankings with ELO-like scores
- Head-to-head record tracking between agent pairs
- Tournament history for analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..env.sim import ParallelRunner, MatchResult, SimulationEngine
from ..env.sim.actions import Action, ActionType
from ..env.sim.state import GameStateSnapshot

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class TournamentFormat(Enum):
    """Supported tournament formats."""
    ROUND_ROBIN = auto()         # All-vs-all
    SINGLE_ELIMINATION = auto()  # Single elimination bracket
    DOUBLE_ELIMINATION = auto()  # Double elimination bracket
    LEAGUE = auto()              # League with promotion/relegation
    SWISS = auto()               # Swiss pairing: N*R/2 matches, scales linearly


# Starting rating for an unseen agent.
DEFAULT_ELO = 1500.0


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class HeadToHeadRecord:
    """Head-to-head record between two agents.

    Attributes:
        agent1_id: First agent ID.
        agent2_id: Second agent ID.
        agent1_wins: Agent 1's wins.
        agent2_wins: Agent 2's wins.
        draws: Draws.
        agent1_towers: Agent 1's total towers destroyed.
        agent2_towers: Agent 2's total towers destroyed.
        agent1_duration: Agent 1's total match duration.
        agent2_duration: Agent 2's total match duration.
        matches: Total matches played.
    """
    agent1_id: str
    agent2_id: str
    agent1_wins: int = 0
    agent2_wins: int = 0
    draws: int = 0
    agent1_towers: int = 0
    agent2_towers: int = 0
    agent1_duration: int = 0
    agent2_duration: int = 0

    @property
    def matches(self) -> int:
        return self.agent1_wins + self.agent2_wins + self.draws

    @property
    def agent1_win_rate(self) -> float:
        if self.matches == 0:
            return 0.0
        return self.agent1_wins / self.matches

    @property
    def agent2_win_rate(self) -> float:
        if self.matches == 0:
            return 0.0
        return self.agent2_wins / self.matches

    @property
    def tower_differential(self) -> float:
        return self.agent1_towers - self.agent2_towers


@dataclass
class AgentTournamentStats:
    """Aggregated tournament stats for a single agent.

    Attributes:
        agent_id: Agent identifier.
        wins: Total wins.
        draws: Total draws.
        losses: Total losses.
        towers_destroyed: Total towers destroyed.
        total_duration: Total match duration.
        elo_rating: ELO-like rating score.
        win_rate: Overall win rate.
        h2h_records: Dict of opponent_id -> HeadToHeadRecord.
    """
    agent_id: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    towers_destroyed: int = 0
    total_duration: int = 0
    elo_rating: float = 1500.0
    _win_rate: float = 0.0
    h2h_records: Dict[str, HeadToHeadRecord] = field(default_factory=dict)

    @property
    def matches(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        """Compute win rate dynamically."""
        if self.matches == 0:
            return self._win_rate
        return self.wins / self.matches

    @property
    def tower_differential(self) -> float:
        return self.towers_destroyed - self.losses

    def compute_composite_score(self, tower_weight: float = 0.1) -> float:
        """Compute a composite score from tournament performance.

        Score = (points per match) + tower_weight * avg tower differential,
        where a win is 1 point, a draw 0.5 and a loss 0.

        Normalising by matches played matters for Swiss pairings, where agents
        can play unequal numbers of games (byes). Match duration is
        deliberately *not* a term: the previous formula added
        ``0.01 * avg_duration``, and with durations in the hundreds of ticks
        that single term outweighed every win in a short tournament, so
        ranking largely reflected which agent happened to play longer games.

        Args:
            tower_weight: Weight for average towers destroyed.

        Returns:
            Composite score.
        """
        matches = max(1, self.matches)
        points = (self.wins * 1.0 + self.draws * 0.5) / matches
        avg_towers = self.towers_destroyed / matches
        return points + tower_weight * avg_towers

    def update_elo(self, opponent_elo: float, actual_score: float,
                   k_factor: float = 32.0) -> None:
        """Update this agent's ELO from a match result.

        Args:
            opponent_elo: Opponent's ELO rating before this match.
            actual_score: This agent's result: 1=win, 0.5=draw, 0=loss.
            k_factor: Maximum rating change per match.
        """
        # The previous implementation derived "actual" from the ratings too
        # (actual = 1 - expected), so the update reduced to k*(1 - 2*expected)
        # and moved ratings without ever consulting the result.
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        self.elo_rating += k_factor * (actual_score - expected)


@dataclass
class TournamentBracket:
    """Representation of a tournament bracket.

    Attributes:
        format: Tournament format used.
        rounds: List of rounds, each containing matchups.
        winners: List of agents that advanced.
        loser_bracket: Agents in loser bracket (for double elimination).
    """
    format: TournamentFormat
    rounds: List[List[Tuple[str, str]]] = field(default_factory=list)
    winners: List[str] = field(default_factory=list)
    loser_bracket: List[str] = field(default_factory=list)


@dataclass
class TournamentResult:
    """Result of a tournament-style evaluation.

    Attributes:
        rankings: List of (agent_id, composite_score) sorted by rank.
        agent_stats: Dict of agent_id -> AgentTournamentStats.
        h2h_records: Dict of (agent1, agent2) -> HeadToHeadRecord.
        wins: Dict of agent_id -> win count.
        draws: Dict of agent_id -> draw count.
        losses: Dict of agent_id -> loss count.
        total_matches: Total matches played.
        bracket: Tournament bracket (for elimination formats).
        elo_ratings: Dict of agent_id -> final ELO rating.
        generation: Generation number this tournament belongs to.
    """
    rankings: List[Tuple[str, float]] = field(default_factory=list)
    agent_stats: Dict[str, AgentTournamentStats] = field(default_factory=dict)
    h2h_records: Dict[Tuple[str, str], HeadToHeadRecord] = field(default_factory=dict)
    wins: Dict[str, int] = field(default_factory=dict)
    draws: Dict[str, int] = field(default_factory=dict)
    losses: Dict[str, int] = field(default_factory=dict)
    total_matches: int = 0
    bracket: Optional[TournamentBracket] = None
    elo_ratings: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    def get_ranking(self, agent_id: str) -> int:
        """Get the ranking position (1-based) of an agent."""
        for i, (aid, _) in enumerate(self.rankings, 1):
            if aid == agent_id:
                return i
        return len(self.rankings) + 1

    def get_win_rate(self, agent_id: str) -> float:
        """Get the win rate of an agent."""
        stats = self.agent_stats.get(agent_id)
        return stats.win_rate if stats else 0.0

    def get_h2h_record(self, agent1_id: str, agent2_id: str) -> Optional[HeadToHeadRecord]:
        """Get head-to-head record between two agents."""
        key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
        return self.h2h_records.get(key)

    def summary(self) -> str:
        """Get a text summary of the tournament results."""
        lines = [f"Tournament Results (Generation {self.generation}):"]
        lines.append(f"Total matches: {self.total_matches}")
        lines.append("")
        lines.append("Rankings:")
        for i, (agent_id, score) in enumerate(self.rankings, 1):
            stats = self.agent_stats.get(agent_id)
            if stats:
                lines.append(
                    f"  {i}. {agent_id}: score={score:.3f}, "
                    f"w={stats.wins}, d={stats.draws}, l={stats.losses}, "
                    f"WR={stats.win_rate:.2%}, ELO={stats.elo_rating:.1f}"
                )
            else:
                lines.append(f"  {i}. {agent_id}: score={score:.3f}")
        return "\n".join(lines)


# =============================================================================
# Tournament Runner
# =============================================================================


class TournamentRunner:
    """Runs tournament-style competitions between agents.

    Supports:
    - Round-robin (all-vs-all)
    - Single elimination bracket
    - Double elimination bracket
    - League format with promotion/relegation

    Uses head-to-head matching with side-swapping for fair evaluation.
    """

    def __init__(self, runner: ParallelRunner, seed: int = 42):
        """Initialize the tournament runner.

        Args:
            runner: ParallelRunner instance for executing matches.
            seed: Random seed.
        """
        self.runner = runner
        self.seed = seed

    # ------------------------------------------------------------------
    # Swiss pairing
    # ------------------------------------------------------------------

    @staticmethod
    def recommended_rounds(num_agents: int) -> int:
        """Swiss rounds needed to resolve a clear winner from ``num_agents``.

        ``ceil(log2(n))`` is the standard choice: each round roughly halves the
        set of plausible leaders.
        """
        if num_agents < 2:
            return 0
        return max(1, int(math.ceil(math.log2(num_agents))))

    @staticmethod
    def _swiss_pair(order: List[int],
                    already_played: set) -> Tuple[List[Tuple[int, int]], Optional[int]]:
        """Pair a standings-ordered list, avoiding rematches where possible.

        Walks the standings from the top, pairing each unpaired agent with the
        nearest-ranked opponent it has not already met. Falls back to the
        nearest available opponent when every candidate is a rematch, so
        pairing always completes.

        Returns:
            ``(pairings, bye)`` where ``bye`` is the odd agent out, if any.
        """
        remaining = list(order)
        pairings: List[Tuple[int, int]] = []

        while len(remaining) >= 2:
            first = remaining.pop(0)
            choice = 0
            for offset, candidate in enumerate(remaining):
                if (min(first, candidate), max(first, candidate)) not in already_played:
                    choice = offset
                    break
            opponent = remaining.pop(choice)
            pairings.append((first, opponent))

        return pairings, (remaining[0] if remaining else None)

    def run_swiss(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 2,
        rounds: Optional[int] = None,
        seed_offset: int = 0,
        initial_elo: Optional[Dict[str, float]] = None,
    ) -> TournamentResult:
        """Run a Swiss-system tournament.

        Every agent plays one matchup per round against an opponent on a
        similar score, so a full field is ranked in ``N*R/2`` matchups rather
        than round-robin's ``N*(N-1)/2``. At a population of 200 that is the
        difference between ~800 matchups and ~19,900, which is what makes
        agent-vs-agent play viable as the main training loop rather than an
        occasional evaluation.

        Args:
            agent_ids: Identifiers, index-aligned with ``weights_list``.
            weights_list: Competitor genomes.
            matches_per_pair: Matches per matchup, split between sides.
            rounds: Swiss rounds. Defaults to ``recommended_rounds``.
            seed_offset: Offset for match seeds.
            initial_elo: Ratings carried in from previous generations. Agents
                start at ``DEFAULT_ELO`` when absent.

        Returns:
            TournamentResult with rankings, per-agent stats, H2H records and
            updated ELO ratings.
        """
        n = len(agent_ids)
        agent_stats = {aid: AgentTournamentStats(agent_id=aid) for aid in agent_ids}
        h2h_records: Dict[Tuple[str, str], HeadToHeadRecord] = {}
        elo_ratings = {aid: (initial_elo or {}).get(aid, DEFAULT_ELO)
                       for aid in agent_ids}

        if n < 2:
            return TournamentResult(
                rankings=[(aid, 0.0) for aid in agent_ids],
                agent_stats=agent_stats, elo_ratings=elo_ratings,
            )

        if rounds is None:
            rounds = self.recommended_rounds(n)

        played: set = set()
        total_matches = 0

        for round_idx in range(rounds):
            # Standings: score first, rating as tie-break. Round 1 has no
            # results yet, so this is the carried-in rating order.
            order = sorted(
                range(n),
                key=lambda i: (agent_stats[agent_ids[i]].compute_composite_score(),
                               elo_ratings[agent_ids[i]]),
                reverse=True,
            )
            pairings, bye = self._swiss_pair(order, played)
            if not pairings:
                break

            results = self.runner.run_pairings(
                pairings=pairings,
                weights_list=weights_list,
                matches_per_pair=matches_per_pair,
                seed=self.seed + seed_offset + round_idx * 10_007,
            )

            for (i, j), result in zip(pairings, results):
                aid1, aid2 = agent_ids[i], agent_ids[j]
                played.add((min(i, j), max(i, j)))
                self._record_matchup(aid1, aid2, result, agent_stats,
                                     h2h_records, elo_ratings)
                total_matches += matches_per_pair

            if bye is not None:
                # A bye is a free half-point, the standard Swiss convention.
                agent_stats[agent_ids[bye]].draws += 1

        return self._finalise(agent_ids, agent_stats, h2h_records,
                              elo_ratings, total_matches)

    def _record_matchup(self, aid1: str, aid2: str, result,
                        agent_stats: Dict[str, AgentTournamentStats],
                        h2h_records: Dict[Tuple[str, str], HeadToHeadRecord],
                        elo_ratings: Dict[str, float]) -> None:
        """Fold one head-to-head result into standings, H2H and ratings."""
        meta = result.metadata
        a1_wins = meta.get("agent1_wins", 0)
        a1_draws = meta.get("agent1_draws", 0)
        a1_losses = meta.get("agent1_losses", 0)
        a1_towers = meta.get("agent1_towers", 0)
        a1_duration = meta.get("agent1_duration", 0)
        a2_wins = meta.get("agent2_wins", 0)
        a2_draws = meta.get("agent2_draws", 0)
        a2_losses = meta.get("agent2_losses", 0)
        a2_towers = meta.get("agent2_towers", 0)
        a2_duration = meta.get("agent2_duration", 0)

        for aid, wins, draws, losses, towers, duration in (
            (aid1, a1_wins, a1_draws, a1_losses, a1_towers, a1_duration),
            (aid2, a2_wins, a2_draws, a2_losses, a2_towers, a2_duration),
        ):
            stats = agent_stats[aid]
            stats.wins += wins
            stats.draws += draws
            stats.losses += losses
            stats.towers_destroyed += towers
            stats.total_duration += duration

        key = (min(aid1, aid2), max(aid1, aid2))
        existing = h2h_records.get(key)
        if existing is None:
            h2h_records[key] = HeadToHeadRecord(
                agent1_id=aid1, agent2_id=aid2,
                agent1_wins=a1_wins, agent2_wins=a2_wins,
                draws=a1_draws + a2_draws,
                agent1_towers=a1_towers, agent2_towers=a2_towers,
                agent1_duration=a1_duration, agent2_duration=a2_duration,
            )
        else:
            existing.agent1_wins += a1_wins
            existing.agent2_wins += a2_wins
            existing.draws += a1_draws + a2_draws

        self._update_elo_pair(aid1, aid2, a1_wins, a1_draws,
                              a2_wins, a2_draws, elo_ratings)

    def _finalise(self, agent_ids: List[str],
                  agent_stats: Dict[str, AgentTournamentStats],
                  h2h_records: Dict[Tuple[str, str], HeadToHeadRecord],
                  elo_ratings: Dict[str, float],
                  total_matches: int) -> TournamentResult:
        """Compute win rates and final rankings."""
        rankings = []
        for aid in agent_ids:
            stats = agent_stats[aid]
            rankings.append((aid, stats.compute_composite_score()))
        rankings.sort(key=lambda pair: pair[1], reverse=True)

        return TournamentResult(
            rankings=rankings,
            agent_stats=agent_stats,
            h2h_records=h2h_records,
            wins={aid: s.wins for aid, s in agent_stats.items()},
            draws={aid: s.draws for aid, s in agent_stats.items()},
            losses={aid: s.losses for aid, s in agent_stats.items()},
            total_matches=total_matches,
            elo_ratings=elo_ratings,
        )

    def run_round_robin(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 4,
        seed_offset: int = 0,
    ) -> TournamentResult:
        """Run a round-robin tournament (all-vs-all).

        Each agent plays against every other agent for the specified
        number of matches, with side-swapping for fairness.

        Args:
            agent_ids: List of agent identifiers.
            weights_list: List of weight arrays corresponding to agent_ids.
            matches_per_pair: Matches per agent pair (split evenly for side-swapping).
            seed_offset: Offset for random seeds.

        Returns:
            TournamentResult with rankings and detailed stats.
        """
        n = len(agent_ids)
        agent_stats = {aid: AgentTournamentStats(agent_id=aid) for aid in agent_ids}
        h2h_records: Dict[Tuple[str, str], HeadToHeadRecord] = {}
        total_matches = 0

        # Initialize ELO ratings
        elo_ratings = {aid: 1500.0 for aid in agent_ids}

        # Round-robin: each pair plays
        for i in range(n):
            for j in range(i + 1, n):
                aid1, aid2 = agent_ids[i], agent_ids[j]
                key = (min(aid1, aid2), max(aid1, aid2))

                # Run head-to-head with side-swapping
                result = self.runner.run_head_to_head(
                    agent1_weights=weights_list[i],
                    agent2_weights=weights_list[j],
                    matches_per_pair=matches_per_pair,
                    seed=self.seed + seed_offset + i * 100 + j,
                )

                # Extract per-agent stats from metadata
                meta = result.metadata
                a1_wins = meta.get("agent1_wins", 0)
                a1_draws = meta.get("agent1_draws", 0)
                a1_losses = meta.get("agent1_losses", 0)
                a1_towers = meta.get("agent1_towers", 0)
                a1_duration = meta.get("agent1_duration", 0)
                a2_wins = meta.get("agent2_wins", 0)
                a2_draws = meta.get("agent2_draws", 0)
                a2_losses = meta.get("agent2_losses", 0)
                a2_towers = meta.get("agent2_towers", 0)
                a2_duration = meta.get("agent2_duration", 0)

                # Update agent stats
                agent_stats[aid1].wins += a1_wins
                agent_stats[aid1].draws += a1_draws
                agent_stats[aid1].losses += a1_losses
                agent_stats[aid1].towers_destroyed += a1_towers
                agent_stats[aid1].total_duration += a1_duration

                agent_stats[aid2].wins += a2_wins
                agent_stats[aid2].draws += a2_draws
                agent_stats[aid2].losses += a2_losses
                agent_stats[aid2].towers_destroyed += a2_towers
                agent_stats[aid2].total_duration += a2_duration

                # Update H2H record
                h2h = HeadToHeadRecord(
                    agent1_id=aid1,
                    agent2_id=aid2,
                    agent1_wins=a1_wins,
                    agent2_wins=a2_wins,
                    draws=a1_draws + a2_draws,
                    agent1_towers=a1_towers,
                    agent2_towers=a2_towers,
                    agent1_duration=a1_duration,
                    agent2_duration=a2_duration,
                )
                h2h_records[key] = h2h

                # Update ELO ratings
                self._update_elo_pair(aid1, aid2, a1_wins, a1_draws, a2_wins, a2_draws,
                                       elo_ratings, k_factor=32.0)

                total_matches += matches_per_pair

        # Compute win rates and composite scores
        rankings = []
        for aid in agent_ids:
            stats = agent_stats[aid]
            # win_rate is a computed property; assigning to it raised
            # AttributeError, which crashed the tail of every tournament
            # format before any ranking was produced.
            score = stats.compute_composite_score()
            rankings.append((aid, score))

        rankings.sort(key=lambda x: x[1], reverse=True)

        return TournamentResult(
            rankings=rankings,
            agent_stats=agent_stats,
            h2h_records=h2h_records,
            wins={aid: stats.wins for aid, stats in agent_stats.items()},
            draws={aid: stats.draws for aid, stats in agent_stats.items()},
            losses={aid: stats.losses for aid, stats in agent_stats.items()},
            total_matches=total_matches,
            elo_ratings=elo_ratings,
        )

    def run_single_elimination(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 4,
        seed_offset: int = 0,
    ) -> TournamentResult:
        """Run a single elimination tournament.

        Agents are seeded by their current fitness (if available).
        Winners advance, losers are eliminated.

        Args:
            agent_ids: List of agent identifiers.
            weights_list: List of weight arrays.
            matches_per_pair: Matches per matchup.
            seed_offset: Offset for random seeds.

        Returns:
            TournamentResult with rankings and bracket.
        """
        n = len(agent_ids)
        agent_stats = {aid: AgentTournamentStats(agent_id=aid) for aid in agent_ids}
        h2h_records: Dict[Tuple[str, str], HeadToHeadRecord] = {}
        total_matches = 0
        elo_ratings = {aid: 1500.0 for aid in agent_ids}

        # Seed: sort by some heuristic (here, alphabetical for determinism)
        # In practice, could use current fitness to seed
        seeded_agents = sorted(agent_ids)  # Could be sorted by fitness

        # Build bracket rounds
        bracket = TournamentBracket(format=TournamentFormat.SINGLE_ELIMINATION)
        current_round = [(seeded_agents[i], seeded_agents[i + 1])
                         for i in range(0, len(seeded_agents) - 1, 2)]
        if len(seeded_agents) % 2 == 1:
            current_round.append((seeded_agents[-1], None))  # Bye

        round_num = 0
        while len(current_round) > 1:
            next_round = []
            for agent1_id, agent2_id in current_round:
                if agent2_id is None:
                    # Bye - agent advances automatically
                    next_round.append(agent1_id)
                    agent_stats[agent1_id].wins += 1
                    continue

                # Get indices
                i1 = agent_ids.index(agent1_id)
                i2 = agent_ids.index(agent2_id)

                # Run head-to-head
                result = self.runner.run_head_to_head(
                    agent1_weights=weights_list[i1],
                    agent2_weights=weights_list[i2],
                    matches_per_pair=matches_per_pair,
                    seed=self.seed + seed_offset + round_num * 1000,
                )

                meta = result.metadata
                a1_wins = meta.get("agent1_wins", 0)
                a1_draws = meta.get("agent1_draws", 0)
                a1_losses = meta.get("agent1_losses", 0)
                a1_towers = meta.get("agent1_towers", 0)
                a1_duration = meta.get("agent1_duration", 0)
                a2_wins = meta.get("agent2_wins", 0)
                a2_draws = meta.get("agent2_draws", 0)
                a2_losses = meta.get("agent2_losses", 0)
                a2_towers = meta.get("agent2_towers", 0)
                a2_duration = meta.get("agent2_duration", 0)

                # Update stats
                agent_stats[agent1_id].wins += a1_wins
                agent_stats[agent1_id].draws += a1_draws
                agent_stats[agent1_id].losses += a1_losses
                agent_stats[agent1_id].towers_destroyed += a1_towers
                agent_stats[agent1_id].total_duration += a1_duration

                agent_stats[agent2_id].wins += a2_wins
                agent_stats[agent2_id].draws += a2_draws
                agent_stats[agent2_id].losses += a2_losses
                agent_stats[agent2_id].towers_destroyed += a2_towers
                agent_stats[agent2_id].total_duration += a2_duration

                # H2H record
                key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
                h2h_records[key] = HeadToHeadRecord(
                    agent1_id=agent1_id, agent2_id=agent2_id,
                    agent1_wins=a1_wins, agent2_wins=a2_wins,
                    draws=a1_draws + a2_draws,
                    agent1_towers=a1_towers, agent2_towers=a2_towers,
                    agent1_duration=a1_duration, agent2_duration=a2_duration,
                )

                # Determine winner (most wins, tiebreak by towers)
                if a1_wins > a2_wins:
                    next_round.append(agent1_id)
                    self._update_elo_pair(agent1_id, agent2_id, a1_wins, a1_draws,
                                          a2_wins, a2_draws, elo_ratings)
                elif a2_wins > a1_wins:
                    next_round.append(agent2_id)
                    self._update_elo_pair(agent2_id, agent1_id, a2_wins, a2_draws,
                                          a1_wins, a1_draws, elo_ratings)
                else:
                    # Tiebreak by towers
                    if a1_towers > a2_towers:
                        next_round.append(agent1_id)
                    else:
                        next_round.append(agent2_id)
                    self._update_elo_pair(agent1_id, agent2_id, a1_draws, a1_draws,
                                          a2_draws, a2_draws, elo_ratings)

                total_matches += matches_per_pair

            bracket.rounds.append(current_round)
            current_round = [(a, None) for a in next_round]
            round_num += 1

        bracket.winners = current_round

        # Compute final rankings
        rankings = []
        for aid in agent_ids:
            stats = agent_stats[aid]
            # win_rate is a computed property; assigning to it raised
            # AttributeError, which crashed the tail of every tournament
            # format before any ranking was produced.
            score = stats.compute_composite_score()
            rankings.append((aid, score))
        rankings.sort(key=lambda x: x[1], reverse=True)

        return TournamentResult(
            rankings=rankings,
            agent_stats=agent_stats,
            h2h_records=h2h_records,
            wins={aid: stats.wins for aid, stats in agent_stats.items()},
            draws={aid: stats.draws for aid, stats in agent_stats.items()},
            losses={aid: stats.losses for aid, stats in agent_stats.items()},
            total_matches=total_matches,
            bracket=bracket,
            elo_ratings=elo_ratings,
        )

    def run_double_elimination(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 4,
        seed_offset: int = 0,
    ) -> TournamentResult:
        """Run a double elimination tournament.

        Agents are eliminated after two losses. The winner of the
        loser bracket final plays the winner bracket final winner
        (with a possible bracket reset).

        Args:
            agent_ids: List of agent identifiers.
            weights_list: List of weight arrays.
            matches_per_pair: Matches per matchup.
            seed_offset: Offset for random seeds.

        Returns:
            TournamentResult with rankings and bracket.
        """
        n = len(agent_ids)
        agent_stats = {aid: AgentTournamentStats(agent_id=aid) for aid in agent_ids}
        h2h_records: Dict[Tuple[str, str], HeadToHeadRecord] = {}
        total_matches = 0
        elo_ratings = {aid: 1500.0 for aid in agent_ids}

        # Initialize brackets
        winner_bracket = list(sorted(agent_ids))
        loser_bracket: List[str] = []
        bracket = TournamentBracket(format=TournamentFormat.DOUBLE_ELIMINATION)
        bracket.loser_bracket = list(loser_bracket)

        round_num = 0
        grand_final_played = False

        while len(winner_bracket) + len(loser_bracket) > 1:
            # Process winner bracket matches
            next_winner = []
            next_loser = []

            for i in range(0, len(winner_bracket) - 1, 2):
                agent1_id = winner_bracket[i]
                agent2_id = winner_bracket[i + 1]

                i1 = agent_ids.index(agent1_id)
                i2 = agent_ids.index(agent2_id)

                result = self.runner.run_head_to_head(
                    agent1_weights=weights_list[i1],
                    agent2_weights=weights_list[i2],
                    matches_per_pair=matches_per_pair,
                    seed=self.seed + seed_offset + round_num * 1000,
                )

                meta = result.metadata
                a1_wins = meta.get("agent1_wins", 0)
                a1_draws = meta.get("agent1_draws", 0)
                a1_losses = meta.get("agent1_losses", 0)
                a1_towers = meta.get("agent1_towers", 0)
                a1_duration = meta.get("agent1_duration", 0)
                a2_wins = meta.get("agent2_wins", 0)
                a2_draws = meta.get("agent2_draws", 0)
                a2_losses = meta.get("agent2_losses", 0)
                a2_towers = meta.get("agent2_towers", 0)
                a2_duration = meta.get("agent2_duration", 0)

                # Update stats
                agent_stats[agent1_id].wins += a1_wins
                agent_stats[agent1_id].draws += a1_draws
                agent_stats[agent1_id].losses += a1_losses
                agent_stats[agent1_id].towers_destroyed += a1_towers
                agent_stats[agent1_id].total_duration += a1_duration

                agent_stats[agent2_id].wins += a2_wins
                agent_stats[agent2_id].draws += a2_draws
                agent_stats[agent2_id].losses += a2_losses
                agent_stats[agent2_id].towers_destroyed += a2_towers
                agent_stats[agent2_id].total_duration += a2_duration

                key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
                h2h_records[key] = HeadToHeadRecord(
                    agent1_id=agent1_id, agent2_id=agent2_id,
                    agent1_wins=a1_wins, agent2_wins=a2_wins,
                    draws=a1_draws + a2_draws,
                    agent1_towers=a1_towers, agent2_towers=a2_towers,
                    agent1_duration=a1_duration, agent2_duration=a2_duration,
                )

                if a1_wins > a2_wins:
                    next_winner.append(agent1_id)
                    next_loser.append(agent2_id)
                elif a2_wins > a1_wins:
                    next_winner.append(agent2_id)
                    next_loser.append(agent1_id)
                else:
                    if a1_towers > a2_towers:
                        next_winner.append(agent1_id)
                        next_loser.append(agent2_id)
                    else:
                        next_winner.append(agent2_id)
                        next_loser.append(agent1_id)

                total_matches += matches_per_pair

            # Handle bye in winner bracket
            if len(winner_bracket) % 2 == 1:
                next_winner.append(winner_bracket[-1])

            # Move losers to loser bracket
            loser_bracket.extend(next_loser)
            winner_bracket = next_winner

            bracket.rounds.append([(winner_bracket[i], winner_bracket[i + 1])
                                    for i in range(0, len(winner_bracket) - 1, 2)])
            bracket.loser_bracket = list(loser_bracket)
            round_num += 1

        # Final: winner bracket champion vs loser bracket champion
        if winner_bracket and loser_bracket:
            wb_champ = winner_bracket[0]
            lb_champ = loser_bracket[0]

            i1 = agent_ids.index(wb_champ)
            i2 = agent_ids.index(lb_champ)

            result = self.runner.run_head_to_head(
                agent1_weights=weights_list[i1],
                agent2_weights=weights_list[i2],
                matches_per_pair=matches_per_pair,
                seed=self.seed + seed_offset + round_num * 1000,
            )

            meta = result.metadata
            a1_wins = meta.get("agent1_wins", 0)
            a1_draws = meta.get("agent1_draws", 0)
            a1_losses = meta.get("agent1_losses", 0)
            a1_towers = meta.get("agent1_towers", 0)
            a1_duration = meta.get("agent1_duration", 0)
            a2_wins = meta.get("agent2_wins", 0)
            a2_draws = meta.get("agent2_draws", 0)
            a2_losses = meta.get("agent2_losses", 0)
            a2_towers = meta.get("agent2_towers", 0)
            a2_duration = meta.get("agent2_duration", 0)

            agent_stats[wb_champ].wins += a1_wins
            agent_stats[wb_champ].draws += a1_draws
            agent_stats[wb_champ].losses += a1_losses
            agent_stats[wb_champ].towers_destroyed += a1_towers
            agent_stats[wb_champ].total_duration += a1_duration

            agent_stats[lb_champ].wins += a2_wins
            agent_stats[lb_champ].draws += a2_draws
            agent_stats[lb_champ].losses += a2_losses
            agent_stats[lb_champ].towers_destroyed += a2_towers
            agent_stats[lb_champ].total_duration += a2_duration

            key = (min(wb_champ, lb_champ), max(wb_champ, lb_champ))
            h2h_records[key] = HeadToHeadRecord(
                agent1_id=wb_champ, agent2_id=lb_champ,
                agent1_wins=a1_wins, agent2_wins=a2_wins,
                draws=a1_draws + a2_draws,
                agent1_towers=a1_towers, agent2_towers=a2_towers,
                agent1_duration=a1_duration, agent2_duration=a2_duration,
            )

            total_matches += matches_per_pair

        # Compute final rankings
        rankings = []
        for aid in agent_ids:
            stats = agent_stats[aid]
            # win_rate is a computed property; assigning to it raised
            # AttributeError, which crashed the tail of every tournament
            # format before any ranking was produced.
            score = stats.compute_composite_score()
            rankings.append((aid, score))
        rankings.sort(key=lambda x: x[1], reverse=True)

        return TournamentResult(
            rankings=rankings,
            agent_stats=agent_stats,
            h2h_records=h2h_records,
            wins={aid: stats.wins for aid, stats in agent_stats.items()},
            draws={aid: stats.draws for aid, stats in agent_stats.items()},
            losses={aid: stats.losses for aid, stats in agent_stats.items()},
            total_matches=total_matches,
            bracket=bracket,
            elo_ratings=elo_ratings,
        )

    def run_league(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 4,
        rounds: int = 3,
        seed_offset: int = 0,
    ) -> TournamentResult:
        """Run a league-style tournament with multiple rounds.

        Agents play each other in each round, with standings
        determining future matchups.

        Args:
            agent_ids: List of agent identifiers.
            weights_list: List of weight arrays.
            matches_per_pair: Matches per matchup per round.
            rounds: Number of rounds to play.
            seed_offset: Offset for random seeds.

        Returns:
            TournamentResult with rankings and bracket.
        """
        agent_stats = {aid: AgentTournamentStats(agent_id=aid) for aid in agent_ids}
        h2h_records: Dict[Tuple[str, str], HeadToHeadRecord] = {}
        total_matches = 0
        elo_ratings = {aid: 1500.0 for aid in agent_ids}

        bracket = TournamentBracket(format=TournamentFormat.LEAGUE)
        standings_history = []

        for round_num in range(rounds):
            # Re-sort by current standings
            sorted_agents = sorted(
                agent_ids,
                key=lambda aid: agent_stats[aid].compute_composite_score(),
                reverse=True,
            )

            round_matches = []
            for i in range(0, len(sorted_agents) - 1, 2):
                agent1_id = sorted_agents[i]
                agent2_id = sorted_agents[i + 1]
                round_matches.append((agent1_id, agent2_id))

                i1 = agent_ids.index(agent1_id)
                i2 = agent_ids.index(agent2_id)

                result = self.runner.run_head_to_head(
                    agent1_weights=weights_list[i1],
                    agent2_weights=weights_list[i2],
                    matches_per_pair=matches_per_pair,
                    seed=self.seed + seed_offset + round_num * 10000 + i,
                )

                meta = result.metadata
                a1_wins = meta.get("agent1_wins", 0)
                a1_draws = meta.get("agent1_draws", 0)
                a1_losses = meta.get("agent1_losses", 0)
                a1_towers = meta.get("agent1_towers", 0)
                a1_duration = meta.get("agent1_duration", 0)
                a2_wins = meta.get("agent2_wins", 0)
                a2_draws = meta.get("agent2_draws", 0)
                a2_losses = meta.get("agent2_losses", 0)
                a2_towers = meta.get("agent2_towers", 0)
                a2_duration = meta.get("agent2_duration", 0)

                agent_stats[agent1_id].wins += a1_wins
                agent_stats[agent1_id].draws += a1_draws
                agent_stats[agent1_id].losses += a1_losses
                agent_stats[agent1_id].towers_destroyed += a1_towers
                agent_stats[agent1_id].total_duration += a1_duration

                agent_stats[agent2_id].wins += a2_wins
                agent_stats[agent2_id].draws += a2_draws
                agent_stats[agent2_id].losses += a2_losses
                agent_stats[agent2_id].towers_destroyed += a2_towers
                agent_stats[agent2_id].total_duration += a2_duration

                key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
                if key not in h2h_records:
                    h2h_records[key] = HeadToHeadRecord(
                        agent1_id=agent1_id, agent2_id=agent2_id,
                    )
                h2h = h2h_records[key]
                h2h.agent1_wins += a1_wins
                h2h.agent2_wins += a2_wins
                h2h.draws += a1_draws + a2_draws
                h2h.agent1_towers += a1_towers
                h2h.agent2_towers += a2_towers
                h2h.agent1_duration += a1_duration
                h2h.agent2_duration += a2_duration

                total_matches += matches_per_pair

            bracket.rounds.append(round_matches)
            standings_history.append(
                sorted(agent_ids,
                       key=lambda aid: agent_stats[aid].compute_composite_score(),
                       reverse=True)
            )

        # Compute final rankings
        rankings = []
        for aid in agent_ids:
            stats = agent_stats[aid]
            # win_rate is a computed property; assigning to it raised
            # AttributeError, which crashed the tail of every tournament
            # format before any ranking was produced.
            score = stats.compute_composite_score()
            rankings.append((aid, score))
        rankings.sort(key=lambda x: x[1], reverse=True)

        return TournamentResult(
            rankings=rankings,
            agent_stats=agent_stats,
            h2h_records=h2h_records,
            wins={aid: stats.wins for aid, stats in agent_stats.items()},
            draws={aid: stats.draws for aid, stats in agent_stats.items()},
            losses={aid: stats.losses for aid, stats in agent_stats.items()},
            total_matches=total_matches,
            bracket=bracket,
            elo_ratings=elo_ratings,
        )

    def _update_elo_pair(self, agent1_id: str, agent2_id: str,
                         a1_wins: int, a1_draws: int,
                         a2_wins: int, a2_draws: int,
                         elo_ratings: Dict[str, float],
                         k_factor: float = 32.0) -> None:
        """Update ELO ratings for both agents after a head-to-head matchup.

        The pair's score is normalised by the number of games actually played
        between them. The previous divisor (``wins + draws + 1``) neither
        counted losses nor summed to 1 across the pair, so a clean 2-0 sweep
        scored 0.67 rather than 1.0 and ratings drifted toward the middle.
        """
        games = a1_wins + a2_wins + a1_draws + a2_draws
        if games <= 0:
            return

        a1_score = (a1_wins + 0.5 * a1_draws) / games
        a2_score = 1.0 - a1_score

        elo1 = elo_ratings.get(agent1_id, DEFAULT_ELO)
        elo2 = elo_ratings.get(agent2_id, DEFAULT_ELO)

        expected1 = 1.0 / (1.0 + 10 ** ((elo2 - elo1) / 400))
        expected2 = 1.0 - expected1

        elo_ratings[agent1_id] = elo1 + k_factor * (a1_score - expected1)
        elo_ratings[agent2_id] = elo2 + k_factor * (a2_score - expected2)

    def run_tournament(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        format: TournamentFormat = TournamentFormat.SWISS,
        matches_per_pair: int = 4,
        seed_offset: int = 0,
        rounds: Optional[int] = None,
        initial_elo: Optional[Dict[str, float]] = None,
    ) -> TournamentResult:
        """Run a tournament of the specified format.

        Args:
            agent_ids: List of agent identifiers.
            weights_list: List of weight arrays.
            format: Tournament format to use. Swiss is the default because it
                is the only one that scales to a full training population.
            matches_per_pair: Matches per matchup.
            seed_offset: Offset for random seeds.
            rounds: Number of rounds (Swiss and league formats).
            initial_elo: Ratings carried in from previous generations.

        Returns:
            TournamentResult with rankings and detailed stats.
        """
        if format == TournamentFormat.SWISS:
            return self.run_swiss(agent_ids, weights_list, matches_per_pair,
                                  rounds, seed_offset, initial_elo)
        if format == TournamentFormat.ROUND_ROBIN:
            return self.run_round_robin(agent_ids, weights_list, matches_per_pair, seed_offset)
        if format == TournamentFormat.SINGLE_ELIMINATION:
            return self.run_single_elimination(agent_ids, weights_list, matches_per_pair, seed_offset)
        if format == TournamentFormat.DOUBLE_ELIMINATION:
            return self.run_double_elimination(agent_ids, weights_list, matches_per_pair, seed_offset)
        if format == TournamentFormat.LEAGUE:
            return self.run_league(agent_ids, weights_list, matches_per_pair,
                                   rounds if rounds is not None else 3, seed_offset)
        raise ValueError(f"Unknown tournament format: {format}")


# =============================================================================
# Fitness Evaluator (Enhanced)
# =============================================================================


class FitnessEvaluator:
    """Evaluates agent fitness against various opponents.

    Supports:
    - Single agent evaluation
    - Population evaluation
    - Tournament-style competitions with multiple formats
    - Head-to-head matchups with side-swapping
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
        self.tournament_runner = TournamentRunner(self.runner)

    def evaluate_against_random(
        self,
        weights: np.ndarray,
        matches: int = 10,
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent against random opponents."""
        result = self.runner.evaluate_single(
            weights=weights, matches=matches, opponent_type="random", seed=seed,
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
        """Evaluate an agent against greedy opponents."""
        result = self.runner.evaluate_single(
            weights=weights, matches=matches, opponent_type="greedy", seed=seed,
        )
        logger.info(f"Evaluated agent vs greedy: fitness={result.fitness:.3f}")
        return result

    def evaluate_against_aggressive(
        self,
        weights: np.ndarray,
        matches: int = 10,
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent against aggressive opponents."""
        result = self.runner.evaluate_single(
            weights=weights, matches=matches, opponent_type="aggressive", seed=seed,
        )
        logger.info(f"Evaluated agent vs aggressive: fitness={result.fitness:.3f}")
        return result

    def evaluate_against_defensive(
        self,
        weights: np.ndarray,
        matches: int = 10,
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent against defensive opponents."""
        result = self.runner.evaluate_single(
            weights=weights, matches=matches, opponent_type="defensive", seed=seed,
        )
        logger.info(f"Evaluated agent vs defensive: fitness={result.fitness:.3f}")
        return result

    def evaluate_self_play(
        self,
        weights: np.ndarray,
        matches: int = 10,
        opponent_weights: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent in self-play mode."""
        result = self.runner.evaluate_single(
            weights=weights, matches=matches, opponent_type="self_play",
            opponent_weights=opponent_weights, seed=seed,
        )
        logger.info(f"Evaluated agent in self-play: fitness={result.fitness:.3f}")
        return result

    def evaluate_against_all_types(
        self,
        weights: np.ndarray,
        matches: int = 10,
        seed: int = 42,
    ) -> Dict[str, MatchResult]:
        """Evaluate an agent against all opponent types."""
        results = {}
        for opp_type in ["random", "greedy", "aggressive", "defensive"]:
            results[opp_type] = self.evaluate_against_opponent(
                weights, matches, opp_type, seed
            )
        return results

    def evaluate_against_opponent(
        self,
        weights: np.ndarray,
        matches: int = 10,
        opponent_type: str = "random",
        seed: int = 42,
    ) -> MatchResult:
        """Evaluate an agent against a specific opponent type."""
        result = self.runner.evaluate_single(
            weights=weights, matches=matches, opponent_type=opponent_type, seed=seed,
        )
        logger.info(f"Evaluated agent vs {opponent_type}: fitness={result.fitness:.3f}, "
                    f"w={result.wins}, d={result.draws}, l={result.losses}")
        return result

    def evaluate_population(
        self,
        weights_list: List[np.ndarray],
        matches_per_agent: int = 5,
        opponent_type: str = "random",
        seed: int = 42,
    ) -> List[MatchResult]:
        """Evaluate a population of agents."""
        results = self.runner.evaluate_population(
            population_weights=weights_list,
            matches_per_agent=matches_per_agent,
            opponent_type=opponent_type,
            seed=seed,
        )
        return results

    def run_head_to_head(
        self,
        agent1_weights: np.ndarray,
        agent2_weights: np.ndarray,
        matches_per_pair: int = 4,
        seed: int = 42,
    ) -> MatchResult:
        """Run a head-to-head matchup between two agents.

        Each agent plays half as player and half as opponent for fairness.

        Args:
            agent1_weights: Weights for agent 1.
            agent2_weights: Weights for agent 2.
            matches_per_pair: Total matches to play.
            seed: Random seed.

        Returns:
            MatchResult with combined stats and per-agent breakdown.
        """
        return self.runner.run_head_to_head(
            agent1_weights=agent1_weights,
            agent2_weights=agent2_weights,
            matches_per_pair=matches_per_pair,
            seed=seed,
        )

    def run_tournament(
        self,
        agent_ids: List[str],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 4,
        seed: int = 42,
        format: TournamentFormat = TournamentFormat.SWISS,
        rounds: Optional[int] = None,
        generation: int = 0,
        initial_elo: Optional[Dict[str, float]] = None,
    ) -> TournamentResult:
        """Run a tournament-style evaluation where agents play each other.

        Uses head-to-head matchups with side-swapping for fairness.
        Supports round-robin, single/double elimination, and league formats.

        Args:
            agent_ids: List of agent identifiers.
            weights_list: List of weight arrays for each agent.
            matches_per_pair: Matches per agent pair per round.
            seed: Random seed.
            format: Tournament format (ROUND_ROBIN, SINGLE_ELIMINATION,
                   DOUBLE_ELIMINATION, LEAGUE).
            rounds: Number of rounds (for league format).
            generation: Generation number for tracking.

        Returns:
            TournamentResult with rankings, ELO ratings, and H2H records.
        """
        logger.info(f"Running {format.name} tournament with {len(agent_ids)} agents, "
                    f"{matches_per_pair} matches per pair")

        result = self.tournament_runner.run_tournament(
            agent_ids=agent_ids,
            weights_list=weights_list,
            format=format,
            matches_per_pair=matches_per_pair,
            seed_offset=seed,
            rounds=rounds,
            initial_elo=initial_elo,
        )
        result.generation = generation

        logger.info(f"Tournament complete: {result.summary()}")
        return result

    def compute_fitness_score(
        self,
        match_result: MatchResult,
        scoring_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute a composite fitness score from match results."""
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
