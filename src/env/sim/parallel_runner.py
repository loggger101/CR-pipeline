"""Parallel simulation runner for training.

Manages multiple independent simulation instances running concurrently,
collecting fitness scores and managing worker lifecycle.

Uses multiprocessing for true parallelism across CPU cores.

Enhanced with:
- Proper agent inference integration
- Multiple opponent AI strategies (random, greedy, balanced)
- Better match result aggregation
- Worker process management
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import queue
import time
import sys
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .engine import SimulationEngine, SimulationStepResult, UnitState, UnitStatus, OpponentStrategy
from .actions import Action, ActionType

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
        avg_elixir_used: Average elixir efficiency.
        metadata: Additional match statistics.
    """
    agent_id: str
    fitness: float
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_towers_destroyed: float = 0.0
    avg_duration: float = 0.0
    avg_elixir_used: float = 0.0
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
        opponent_type: Type of opponent ("random", "greedy", "balanced", 
                       "aggressive", "defensive", "self_play").
        self_play_weights: Weights for self-play opponent.
        use_training_decks: Whether to use varied training decks for opponents.
    """
    seed: int = 42
    match_count: int = 5
    deck: Optional[List[str]] = None
    opponent_deck: Optional[List[str]] = None
    match_duration_ticks: int = 1800
    overtime_ticks: int = 120
    elixir_regen_rate: float = 0.3
    opponent_type: str = "random"
    self_play_weights: Optional[np.ndarray] = None
    use_training_decks: bool = True


def _get_card_def(card_name: str):
    """Get card definition by name."""
    from .entities import CARD_DEFS
    return CARD_DEFS.get(card_name)


def _affordable_slots(engine, side: str) -> np.ndarray:
    """Boolean mask of hand slots that can legally be played right now."""
    hand = engine.player_hand if side == "player" else engine.opponent_hand
    cooldowns = (engine.player_cooldowns if side == "player"
                 else engine.opponent_cooldowns)
    elixir = engine.player_elixir if side == "player" else engine.opponent_elixir

    valid = np.zeros(len(hand), dtype=bool)
    for i, card_name in enumerate(hand):
        if cooldowns[i] > 0:
            continue
        card_def = _get_card_def(card_name)
        if card_def is not None and elixir >= card_def.cost:
            valid[i] = True
    return valid


def _placement_to_arena(engine, placement, card_def, side: str):
    """Map the policy's [-1, 1] placement outputs onto legal arena coordinates.

    Units are confined to the acting side's half; spells may target anywhere.
    Coordinates are produced in engine space (row 0 is the opponent's back
    line) regardless of which side is acting.
    """
    col = float(np.clip((placement[0] + 1.0) / 2.0 * (engine.GRID_COLS - 1),
                        0, engine.GRID_COLS - 1))

    is_spell = card_def is not None and card_def.card_type == "spell"
    frac = (placement[1] + 1.0) / 2.0  # 0 = enemy back line, 1 = own back line

    if is_spell:
        lo, hi = 0.0, float(engine.GRID_ROWS - 1)
    elif side == "player":
        lo, hi = float(engine.BRIDGE_ROW + 1), float(engine.GRID_ROWS - 1)
    else:
        lo, hi = 0.0, float(engine.BRIDGE_ROW)

    if side == "player":
        row = lo + frac * (hi - lo)
    else:
        # Mirror: the opponent's "forward" is increasing row.
        row = hi - frac * (hi - lo)

    return col, float(np.clip(row, lo, hi))


def _policy_action(genome, engine, side: str) -> Action:
    """Choose an action for ``side`` by running the evolved policy.

    This replaces a stub that seeded an RNG from the tick counter and never
    read ``genome`` at all -- meaning every agent in the population behaved
    identically and fitness carried no information about the genome.

    Selection is deterministic given the genome and the engine state, so a
    given (genome, seed) pair always produces the same match. Exploration is
    the evolutionary operator's job, not the evaluator's.
    """
    if genome is None:
        return Action.pass_action()

    from ...models.policy import DEFAULT_POLICY_SPEC, encode_features, policy_forward

    valid = _affordable_slots(engine, side)
    features = encode_features(engine, side)
    card_logits, placement = policy_forward(genome, features, DEFAULT_POLICY_SPEC)

    # Slot 4 is "pass" and is always available: holding elixir is a real move.
    masked = card_logits.astype(np.float64).copy()
    masked[:len(valid)][~valid] = -np.inf

    choice = int(np.argmax(masked))
    if choice >= len(valid) or not valid[choice]:
        return Action.pass_action()

    hand = engine.player_hand if side == "player" else engine.opponent_hand
    card_def = _get_card_def(hand[choice])
    col, row = _placement_to_arena(engine, placement, card_def, side)

    action_type = (ActionType.DEPLOY_SPELL
                   if card_def is not None and card_def.card_type == "spell"
                   else ActionType.DEPLOY_UNIT)
    return Action(action_type=action_type, card_index=choice,
                  target_col=col, target_row=row)


def _select_agent_action(state, weights, engine, opponent_type, opponent_weights):
    """Select the player's action for this tick.

    Args:
        state: Current GameStateSnapshot (unused; kept for call compatibility).
        weights: The agent's genome.
        engine: SimulationEngine instance for game state access.
        opponent_type: Type of opponent (unused here).
        opponent_weights: Weights for elite opponents (unused here).

    Returns:
        Action to execute.
    """
    return _policy_action(weights, engine, "player")


@dataclass(frozen=True)
class OpponentProfile:
    """Personality knobs for the scripted heuristic opponent.

    The previous opponents picked a card by cost and then dropped it at a
    uniformly random column and row. They never defended a lane and never
    responded to a push, so an untrained random genome beat all of them 75-100%
    of the time -- fitness hit its ceiling within a generation or two and
    selection had almost nothing left to sort on.
    """
    name: str
    defends: bool = True            # Answer pushes that cross the river
    pushes: bool = True             # Start attacks of its own
    push_at_elixir: float = 7.0     # Elixir at which to commit to a push
    min_play_gap: int = 4           # Ticks to wait between plays
    threat_row: float = 4.0         # React to player troops at/below this row
    uses_spells: bool = True        # Answer clumps with a damage spell
    spell_min_targets: int = 3      # Troops needed before a spell is worth it
    jitter: float = 0.0             # Random placement noise, in tiles
    prefer_cheap_defence: bool = True


OPPONENT_PROFILES = {
    # Deliberately weak control: plays legal cards at arbitrary spots.
    "random": OpponentProfile(
        name="random", defends=False, pushes=True, push_at_elixir=0.0,
        min_play_gap=6, uses_spells=False, jitter=3.0,
        prefer_cheap_defence=False),
    # Elixir dumper: commits its biggest affordable card as soon as it can.
    "greedy": OpponentProfile(
        name="greedy", defends=False, pushes=True, push_at_elixir=0.0,
        min_play_gap=4, uses_spells=False, jitter=0.5,
        prefer_cheap_defence=False),
    "balanced": OpponentProfile(
        name="balanced", defends=True, pushes=True, push_at_elixir=7.0,
        min_play_gap=4, jitter=0.3),
    "aggressive": OpponentProfile(
        name="aggressive", defends=True, pushes=True, push_at_elixir=5.0,
        min_play_gap=3, threat_row=2.5, jitter=0.3,
        prefer_cheap_defence=False),
    "defensive": OpponentProfile(
        name="defensive", defends=True, pushes=True, push_at_elixir=9.5,
        min_play_gap=5, threat_row=4.5, jitter=0.2),
}


def _ticks_since_opponent_played(engine, lookback: int = 12) -> int:
    """Ticks elapsed since the opponent last successfully deployed a card."""
    history = engine.action_history
    for record in reversed(history[-lookback:]):
        if record["player"] == "opponent":
            return engine.tick - record["tick"]
    return 10 ** 6


def _defensive_value(card_def, threat) -> float:
    """Score a card as an answer to ``threat`` (higher is better).

    Approximates staying power times damage output per elixir, which is what
    decides whether a defensive placement trades up or down.
    """
    if threat.is_air and not card_def.target_air:
        return -1.0
    if not threat.is_air and not card_def.target_ground:
        return -1.0
    dps = card_def.damage / max(card_def.attack_speed, 0.1)
    return (dps * card_def.hitpoints) / max(card_def.cost, 0.5) ** 2


def _push_value(card_def) -> float:
    """Score a card as the spearhead of an attack."""
    return card_def.hitpoints + card_def.damage * 2.0


def _biggest_threat(engine, profile):
    """The player troop furthest into the opponent's half, if any.

    Player troops advance towards row 0, so "most advanced" is the smallest
    row.
    """
    best = None
    for unit in engine.player_units:
        if not unit.is_alive or unit.is_building:
            continue
        if unit.row > profile.threat_row:
            continue
        if best is None or unit.row < best.row:
            best = unit
    return best


def _spell_opportunity(engine, hand_slots, profile):
    """Find a damage spell worth casting on a clump of player troops."""
    troops = [u for u in engine.player_units
              if u.is_alive and not u.is_building]
    if len(troops) < profile.spell_min_targets:
        return None

    for slot in hand_slots:
        card_def = _get_card_def(engine.opponent_hand[slot])
        if card_def is None or card_def.card_type != "spell":
            continue
        if card_def.spell_damage <= 0 or card_def.spell_radius <= 0:
            continue
        # Centre on the densest cluster this spell could cover.
        best_centre, best_hits = None, 0
        for candidate in troops:
            hits = sum(
                1 for other in troops
                if math.hypot(other.col - candidate.col,
                              other.row - candidate.row) <= card_def.spell_radius
            )
            if hits > best_hits:
                best_centre, best_hits = candidate, hits
        if best_centre is not None and best_hits >= profile.spell_min_targets:
            return Action(action_type=ActionType.DEPLOY_SPELL,
                          card_index=slot,
                          target_col=float(best_centre.col),
                          target_row=float(best_centre.row))
    return None


def _clamp_opponent_placement(engine, col: float, row: float):
    """Clamp a placement into the opponent's legal deployment half."""
    col = float(np.clip(col, 0.0, engine.GRID_COLS - 1))
    row = float(np.clip(row, 0.0, float(engine.BRIDGE_ROW)))
    return col, row


def _heuristic_opponent_action(engine, profile: OpponentProfile) -> Action:
    """Pick an action for the scripted opponent.

    Order of business: answer a push that has crossed the river, otherwise
    commit to an attack once elixir has built up, otherwise hold. Holding
    elixir is a real choice -- the old opponents spent theirs on the first
    affordable card every few ticks and were permanently behind on tempo.
    """
    rng = engine.rng

    if _ticks_since_opponent_played(engine) < profile.min_play_gap:
        return Action.pass_action()

    valid = _affordable_slots(engine, "opponent")
    slots = [i for i, ok in enumerate(valid) if ok]
    if not slots:
        return Action.pass_action()

    unit_slots = [
        i for i in slots
        if (_get_card_def(engine.opponent_hand[i]) or None) is not None
        and _get_card_def(engine.opponent_hand[i]).card_type != "spell"
    ]

    threat = _biggest_threat(engine, profile) if profile.defends else None

    if threat is not None:
        if profile.uses_spells:
            spell = _spell_opportunity(engine, slots, profile)
            if spell is not None:
                return spell

        scored = [
            (_defensive_value(_get_card_def(engine.opponent_hand[i]), threat), i)
            for i in unit_slots
        ]
        scored = [(value, i) for value, i in scored if value > 0]
        if scored:
            scored.sort(reverse=True)
            if profile.prefer_cheap_defence:
                # Among the top answers, take the cheapest to stay elixir-positive.
                top = scored[:2]
                slot = min(
                    top,
                    key=lambda pair: _get_card_def(
                        engine.opponent_hand[pair[1]]).cost,
                )[1]
            else:
                slot = scored[0][1]
            # Place between the threat and the towers it is walking towards.
            col = threat.col + rng.uniform(-profile.jitter, profile.jitter)
            row = threat.row - 0.5
            col, row = _clamp_opponent_placement(engine, col, row)
            return Action(action_type=ActionType.DEPLOY_UNIT, card_index=slot,
                          target_col=col, target_row=row)

    if not profile.pushes or engine.opponent_elixir < profile.push_at_elixir:
        return Action.pass_action()

    if not unit_slots:
        return Action.pass_action()

    slot = max(unit_slots,
               key=lambda i: _push_value(_get_card_def(engine.opponent_hand[i])))

    # Attack the lane whose defending tower is weakest, entering at the bridge.
    towers = [t for t in engine.player_towers if t.is_alive and not t.is_king]
    if towers:
        target_col = min(towers, key=lambda t: t.hp).col
    else:
        target_col = engine.KING_COL
    bridge_col = min(engine.BRIDGE_COLS, key=lambda c: abs(c - target_col))
    col = bridge_col + rng.uniform(-profile.jitter, profile.jitter)
    col, row = _clamp_opponent_placement(engine, col, float(engine.BRIDGE_ROW))
    return Action(action_type=ActionType.DEPLOY_UNIT, card_index=slot,
                  target_col=col, target_row=row)


def _random_opponent_action(engine) -> Action:
    """Weak control opponent: legal plays at arbitrary positions."""
    return _heuristic_opponent_action(engine, OPPONENT_PROFILES["random"])


def _greedy_opponent_action(engine) -> Action:
    """Spends elixir on its biggest affordable card as soon as it can."""
    return _heuristic_opponent_action(engine, OPPONENT_PROFILES["greedy"])


def _balanced_opponent_action(engine) -> Action:
    """Defends pushes, then counter-attacks once elixir has built up."""
    return _heuristic_opponent_action(engine, OPPONENT_PROFILES["balanced"])


def _aggressive_opponent_action(engine) -> Action:
    """Reacts late, commits early, and pushes on a small elixir bank."""
    return _heuristic_opponent_action(engine, OPPONENT_PROFILES["aggressive"])


def _defensive_opponent_action(engine) -> Action:
    """Answers threats cheaply and only attacks on a near-full bank."""
    return _heuristic_opponent_action(engine, OPPONENT_PROFILES["defensive"])


def _self_play_opponent_action(state, weights, engine) -> Action:
    """Generate the opponent's action from a genome (self-play).

    Uses the same policy as the agent, with features encoded from the
    opponent's side of the board.
    """
    return _policy_action(weights, engine, "opponent")


# Scripted opponents, dispatched by name. "self_play" is handled separately
# because it needs the opposing genome.
_OPPONENT_ACTIONS = {
    "random": _random_opponent_action,
    "greedy": _greedy_opponent_action,
    "balanced": _balanced_opponent_action,
    "aggressive": _aggressive_opponent_action,
    "defensive": _defensive_opponent_action,
}


def _run_head_to_head(
    match_id: int,
    agent1_weights: np.ndarray,
    agent2_weights: np.ndarray,
    matches_per_pair: int,
    seed: int = 42,
    deck: Optional[List[str]] = None,
    opponent_deck: Optional[List[str]] = None,
) -> MatchResult:
    """Run head-to-head matches between two agents.

    Each agent plays half the matches as player (bottom) and half as opponent (top).
    This ensures fair evaluation regardless of side advantage.

    Args:
        match_id: Identifier for this matchup.
        agent1_weights: Weights for agent 1 (plays as player in first half).
        agent2_weights: Weights for agent 2 (plays as opponent in first half).
        matches_per_pair: Total matches to play (split evenly between sides).
        seed: Random seed.
        deck: Card deck for agent 1.
        opponent_deck: Card deck for agent 2.

    Returns:
        MatchResult with combined stats for both agents.
    """
    half_matches = max(1, matches_per_pair // 2)

    stats = {
        "agent1": {"wins": 0, "draws": 0, "losses": 0, "towers": 0, "duration": 0},
        "agent2": {"wins": 0, "draws": 0, "losses": 0, "towers": 0, "duration": 0},
    }

    def play_block(bottom_key, bottom_genome, bottom_deck,
                   top_key, top_genome, top_deck, seed_base):
        """Run ``half_matches`` with the given agent on each side.

        Each genome is evaluated through ``_policy_action`` for the side it is
        actually playing. The previous implementation computed both agents'
        actions as the *player* side and then, in the swapped block, still
        passed agent 1 as the player while recording the result as if agent 2
        had been -- so side-swapping both failed to swap and inverted the win
        attribution it was meant to make fair.
        """
        for i in range(half_matches):
            engine = SimulationEngine(
                deck=bottom_deck,
                opponent_deck=top_deck,
                seed=seed_base + i * 100,
                record_replay=False,
            )
            engine.reset()

            max_ticks = engine.match_duration_ticks + engine.overtime_ticks + 100
            while not engine.terminated and engine.tick <= max_ticks:
                engine.step(
                    _policy_action(bottom_genome, engine, "player"),
                    _policy_action(top_genome, engine, "opponent"),
                )

            bottom, top = stats[bottom_key], stats[top_key]
            if engine.player_trophies > engine.opponent_trophies:
                bottom["wins"] += 1
                top["losses"] += 1
            elif engine.opponent_trophies > engine.player_trophies:
                top["wins"] += 1
                bottom["losses"] += 1
            else:
                bottom["draws"] += 1
                top["draws"] += 1

            # "<side>_towers_destroyed" counts towers that side *lost*.
            bottom["towers"] += engine.opponent_towers_destroyed
            top["towers"] += engine.player_towers_destroyed
            bottom["duration"] += engine.tick
            top["duration"] += engine.tick

    # Phase 1: agent1 on the bottom. Phase 2: agent2 on the bottom.
    play_block("agent1", agent1_weights, deck,
               "agent2", agent2_weights, opponent_deck, seed)
    play_block("agent2", agent2_weights, opponent_deck,
               "agent1", agent1_weights, deck, seed + 50000)

    agent1_wins, agent1_draws = stats["agent1"]["wins"], stats["agent1"]["draws"]
    agent1_losses, agent1_towers = stats["agent1"]["losses"], stats["agent1"]["towers"]
    agent1_duration = stats["agent1"]["duration"]
    agent2_wins, agent2_draws = stats["agent2"]["wins"], stats["agent2"]["draws"]
    agent2_losses, agent2_towers = stats["agent2"]["losses"], stats["agent2"]["towers"]
    agent2_duration = stats["agent2"]["duration"]

    # Compute fitness scores
    agent1_fitness = (agent1_wins * 1.0 + agent1_draws * 0.5 - agent1_losses * 0.5 
                     + 0.1 * agent1_towers / half_matches)
    agent2_fitness = (agent2_wins * 1.0 + agent2_draws * 0.5 - agent2_losses * 0.5 
                     + 0.1 * agent2_towers / half_matches)

    logger.info(f"Head-to-head {match_id}: agent1(fitness={agent1_fitness:.3f}, "
                f"w={agent1_wins}, d={agent1_draws}, l={agent1_losses}) vs "
                f"agent2(fitness={agent2_fitness:.3f}, w={agent2_wins}, d={agent2_draws}, l={agent2_losses})")

    return MatchResult(
        agent_id=f"match_{match_id}",
        fitness=(agent1_fitness + agent2_fitness) / 2,
        wins=agent1_wins + agent2_wins,
        draws=agent1_draws + agent2_draws,
        losses=agent1_losses + agent2_losses,
        avg_towers_destroyed=(agent1_towers + agent2_towers) / (2 * half_matches),
        avg_duration=(agent1_duration + agent2_duration) / (2 * half_matches),
        metadata={
            "agent1_wins": agent1_wins, "agent1_draws": agent1_draws, "agent1_losses": agent1_losses,
            "agent2_wins": agent2_wins, "agent2_draws": agent2_draws, "agent2_losses": agent2_losses,
            "agent1_fitness": agent1_fitness, "agent2_fitness": agent2_fitness,
            "agent1_towers": agent1_towers, "agent2_towers": agent2_towers,
            "agent1_duration": agent1_duration, "agent2_duration": agent2_duration,
        },
    )


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
        opponent_type: Type of opponent ("random", "greedy", "balanced", 
                       "aggressive", "defensive", "self_play").
        opponent_weights: Weights for elite/self-play opponents.

    Returns:
        Aggregated MatchResult.
    """
    logger.info(f"Worker {worker_id}: Starting {config.match_count} matches "
                f"(opponent={opponent_type})")

    # Use training decks for opponent if enabled
    # Opponent decks are drawn per match from config.seed alone -- deliberately
    # *not* from worker_id. Every agent in a generation faces the same sequence
    # of archetypes, so fitness differences reflect play rather than draw luck,
    # while spanning several decks keeps a single lucky matchup from deciding
    # the whole generation.
    if config.use_training_decks and config.opponent_deck is None:
        from .engine import SimulationEngine
        deck_names = sorted(SimulationEngine.TRAINING_DECKS.keys())
        deck_rng = np.random.RandomState(config.seed)
        match_decks = [
            SimulationEngine.TRAINING_DECKS[deck_names[deck_rng.randint(len(deck_names))]]
            for _ in range(max(1, config.match_count))
        ]
    else:
        match_decks = [config.opponent_deck] * max(1, config.match_count)

    engine = SimulationEngine(
        deck=config.deck,
        opponent_deck=match_decks[0],
        match_duration_ticks=config.match_duration_ticks,
        overtime_ticks=config.overtime_ticks,
        elixir_regen_rate=config.elixir_regen_rate,
        seed=config.seed,
    )

    wins = 0
    draws = 0
    losses = 0
    total_towers = 0
    total_duration = 0
    total_elixir = 0.0
    total_card_usage = {}

    opponent_fn = _OPPONENT_ACTIONS.get(opponent_type, _random_opponent_action)
    max_ticks = config.match_duration_ticks + config.overtime_ticks + 100

    for match_idx in range(config.match_count):
        # Distinct seed per match (resetting with one seed replayed the same
        # game), but derived only from config.seed so that every agent in a
        # generation plays the *same* set of matches. These common random
        # numbers turn fitness comparison into a paired test: differences
        # reflect play, not which agent drew the kinder opponents.
        state = engine.reset(seed=config.seed + match_idx,
                             opponent_deck=match_decks[match_idx])

        while not engine.terminated:
            agent_action = _select_agent_action(state, weights, engine,
                                                opponent_type, opponent_weights)

            if opponent_type == "self_play":
                opp_action = _self_play_opponent_action(state, opponent_weights, engine)
            else:
                opp_action = opponent_fn(engine)

            # Read the card before stepping: playing it cycles the slot.
            if agent_action.is_deploy_action() and agent_action.card_index is not None:
                card = engine.player_hand[agent_action.card_index]
                total_card_usage[card] = total_card_usage.get(card, 0) + 1

            result = engine.step(agent_action, opp_action)

            # Safety valve against a stalled match.
            if engine.tick > max_ticks:
                break

        # Record results
        if engine.terminated:
            if engine.player_trophies > engine.opponent_trophies:
                wins += 1
            elif engine.opponent_trophies > engine.player_trophies:
                losses += 1
            else:
                draws += 1

            total_towers += engine.opponent_towers_destroyed
            total_duration += engine.tick
            total_elixir += engine.player_elixir

    avg_towers = total_towers / config.match_count if config.match_count > 0 else 0
    avg_dur = total_duration / config.match_count if config.match_count > 0 else 0
    avg_elixir = total_elixir / config.match_count if config.match_count > 0 else 0

    # Compute fitness with shaped rewards
    fitness = (
        wins * 1.0
        + draws * 0.5
        - losses * 0.5
        + 0.1 * avg_towers
        + 0.05 * (avg_elixir / 10.0)  # Elixir efficiency bonus
    )

    logger.info(f"Worker {worker_id}: fitness={fitness:.3f}, "
                f"w={wins}, d={draws}, l={losses}, "
                f"avg_towers={avg_towers:.1f}, avg_dur={avg_dur:.0f}")

    return MatchResult(
        agent_id=f"worker_{worker_id}",
        fitness=fitness,
        wins=wins,
        draws=draws,
        losses=losses,
        avg_towers_destroyed=avg_towers,
        avg_duration=avg_dur,
        avg_elixir_used=avg_elixir,
        metadata={"card_usage": total_card_usage},
    )


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
            opponent_type: Type of opponent ("random", "greedy", "balanced").
            opponent_weights: Weights for elite opponents.
            deck: Card deck to use for evaluation.
            opponent_deck: Opponent card deck.
            seed: Base random seed.

        Returns:
            List of MatchResult for each agent.
        """
        if self.pool is None:
            self.start()

        # Prepare worker tasks. Argument order must match _run_matches:
        # (worker_id, config, weights, opponent_type, opponent_weights)
        tasks = []
        for i, weights in enumerate(population_weights):
            config = WorkerConfig(
                # Same seed for every agent: common random numbers. Offsetting
                # by the agent index gave each agent its own set of matches, so
                # fitness differences were dominated by draw luck rather than
                # by play, and selection largely sorted noise.
                seed=seed,
                match_count=matches_per_agent,
                deck=deck,
                opponent_deck=opponent_deck,
                opponent_type=opponent_type,
            )
            tasks.append((i, config, weights, opponent_type, opponent_weights))

        raw_results = self.pool.starmap(
            _run_matches, tasks, chunksize=max(1, len(tasks) // 4))

        results = []
        for agent_idx, result in enumerate(raw_results):
            result.agent_id = f"agent_{agent_idx}"
            results.append(result)
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
            opponent_type=opponent_type,
        )
        return _run_matches(0, config, weights, opponent_type)

    def run_head_to_head(
        self,
        agent1_weights: np.ndarray,
        agent2_weights: np.ndarray,
        matches_per_pair: int = 4,
        seed: int = 42,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
    ) -> MatchResult:
        """Run head-to-head matches between two agents.

        Each agent plays half as player and half as opponent to ensure fairness.

        Args:
            agent1_weights: Weights for agent 1.
            agent2_weights: Weights for agent 2.
            matches_per_pair: Total matches to play.
            seed: Random seed.
            deck: Card deck for agent 1.
            opponent_deck: Card deck for agent 2.

        Returns:
            MatchResult with combined stats.
        """
        return _run_head_to_head(
            match_id=0,
            agent1_weights=agent1_weights,
            agent2_weights=agent2_weights,
            matches_per_pair=matches_per_pair,
            seed=seed,
            deck=deck,
            opponent_deck=opponent_deck,
        )

    def run_pairings(
        self,
        pairings: List[Tuple[int, int]],
        weights_list: List[np.ndarray],
        matches_per_pair: int = 4,
        seed: int = 42,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
    ) -> List[MatchResult]:
        """Play a batch of head-to-head matchups across the worker pool.

        Tournament rounds are embarrassingly parallel: every pairing in a round
        is independent. Running them one at a time (as the round-robin code
        did) left a whole generation of matchmaking on a single core.

        Args:
            pairings: ``(index_a, index_b)`` pairs into ``weights_list``.
            weights_list: Genomes for every competitor.
            matches_per_pair: Matches per pairing, split between sides.
            seed: Base seed; each pairing derives its own from this.
            deck: Deck for the first agent of each pairing.
            opponent_deck: Deck for the second agent.

        Returns:
            One MatchResult per pairing, in the order given.
        """
        if not pairings:
            return []

        if self.pool is None:
            self.start()

        tasks = [
            (
                idx,
                weights_list[a],
                weights_list[b],
                matches_per_pair,
                # Derived from the pairing so a matchup replays identically
                # regardless of how rounds are scheduled across workers.
                seed + idx * 977,
                deck,
                opponent_deck,
            )
            for idx, (a, b) in enumerate(pairings)
        ]

        return self.pool.starmap(
            _run_head_to_head, tasks, chunksize=max(1, len(tasks) // 8))
