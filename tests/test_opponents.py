"""Tests for the scripted opponents used as training baselines.

Baseline strength sets the ceiling on what training can learn. When every
scripted opponent loses to an untrained genome, fitness saturates in a couple
of generations and selection has nothing left to sort on -- the pipeline looks
healthy while learning almost nothing.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim.engine import SimulationEngine
from src.env.sim.entities import CARD_DEFS
from src.env.sim.parallel_runner import (
    OPPONENT_PROFILES, WorkerConfig, _OPPONENT_ACTIONS, _heuristic_opponent_action,
    _run_matches,
)
from src.models.policy import DEFAULT_POLICY_SPEC

COMPETITIVE = ["balanced", "aggressive", "defensive"]


def _genome(seed: int) -> np.ndarray:
    return DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(seed))


_RECORD_CACHE = {}


def _record_against(opponent, agents=12, matches=3):
    """Aggregate W/D/L for untrained genomes against a scripted opponent.

    Sampled over a spread of genomes and match seeds: a handful of matches
    swings far too much to characterise a baseline. Cached because several
    tests read the same numbers.
    """
    key = (opponent, agents, matches)
    if key in _RECORD_CACHE:
        return _RECORD_CACHE[key]

    wins = draws = losses = 0
    for i in range(agents):
        config = WorkerConfig(seed=101 + i * 37, match_count=matches)
        result = _run_matches(i, config, _genome(2000 + i), opponent, None)
        wins += result.wins
        draws += result.draws
        losses += result.losses

    _RECORD_CACHE[key] = (wins, draws, losses)
    return _RECORD_CACHE[key]


class TestOpponentsPlayLegally:

    @pytest.mark.parametrize("name", sorted(_OPPONENT_ACTIONS))
    def test_only_deploys_on_its_own_half(self, name):
        action_fn = _OPPONENT_ACTIONS[name]
        engine = SimulationEngine(seed=4, record_replay=False)
        engine.reset()

        for _ in range(400):
            if engine.terminated:
                break
            engine.step(None, action_fn(engine))

        plays = [a for a in engine.action_history if a["player"] == "opponent"]
        for play in plays:
            if CARD_DEFS[play["card"]].card_type != "spell":
                assert play["row"] <= engine.BRIDGE_ROW
            assert 0 <= play["col"] <= engine.GRID_COLS - 1

    @pytest.mark.parametrize("name", sorted(_OPPONENT_ACTIONS))
    def test_actually_plays_cards(self, name):
        action_fn = _OPPONENT_ACTIONS[name]
        engine = SimulationEngine(seed=4, record_replay=False)
        engine.reset()

        for _ in range(600):
            if engine.terminated:
                break
            engine.step(None, action_fn(engine))

        plays = [a for a in engine.action_history if a["player"] == "opponent"]
        assert plays, f"{name} never deployed anything"

    @pytest.mark.parametrize("name", sorted(_OPPONENT_ACTIONS))
    def test_never_overspends_elixir(self, name):
        action_fn = _OPPONENT_ACTIONS[name]
        engine = SimulationEngine(seed=6, record_replay=False)
        engine.reset()

        for _ in range(600):
            if engine.terminated:
                break
            engine.step(None, action_fn(engine))
            assert engine.opponent_elixir >= 0.0

    def test_opponents_are_deterministic_for_a_seed(self):
        def run(seed):
            engine = SimulationEngine(seed=seed, record_replay=False)
            engine.reset()
            for _ in range(300):
                if engine.terminated:
                    break
                engine.step(None, _OPPONENT_ACTIONS["balanced"](engine))
            return [(a["tick"], a["card"]) for a in engine.action_history]

        assert run(11) == run(11)


class TestOpponentBehaviour:

    def test_defensive_profile_answers_a_push(self):
        """A troop across the river must draw a response."""
        engine = SimulationEngine(seed=8, record_replay=False)
        engine.reset()
        engine.opponent_elixir = 10.0
        engine._spawn_unit("knight", 3.0, 2.0, "player")  # over the bridge

        action = _heuristic_opponent_action(
            engine, OPPONENT_PROFILES["defensive"])

        assert action.is_deploy_action()
        # Placed to intercept, in the same lane as the threat.
        assert abs(action.target_col - 3.0) <= 1.5

    def test_defensive_profile_holds_when_unthreatened(self):
        """Holding elixir is a real choice; the old opponents dumped it."""
        engine = SimulationEngine(seed=8, record_replay=False)
        engine.reset()
        engine.opponent_elixir = 4.0  # below the defensive push threshold

        action = _heuristic_opponent_action(
            engine, OPPONENT_PROFILES["defensive"])

        assert not action.is_deploy_action()

    def test_a_full_bank_triggers_a_push(self):
        engine = SimulationEngine(seed=8, record_replay=False)
        engine.reset()
        engine.opponent_elixir = 10.0

        action = _heuristic_opponent_action(
            engine, OPPONENT_PROFILES["balanced"])

        assert action.is_deploy_action()

    def test_pushes_enter_through_a_bridge(self):
        engine = SimulationEngine(seed=8, record_replay=False)
        engine.reset()
        engine.opponent_elixir = 10.0

        action = _heuristic_opponent_action(
            engine, OPPONENT_PROFILES["balanced"])

        nearest_bridge = min(engine.BRIDGE_COLS,
                             key=lambda c: abs(c - action.target_col))
        assert abs(action.target_col - nearest_bridge) <= 1.0

    def test_respects_its_minimum_play_gap(self):
        """Without a gap the opponent chain-plays and wastes its bank."""
        engine = SimulationEngine(seed=8, record_replay=False)
        engine.reset()
        engine.opponent_elixir = 10.0
        engine.action_history.append(
            {"tick": engine.tick, "player": "opponent", "card": "knight",
             "col": 3.0, "row": 2.0})

        action = _heuristic_opponent_action(
            engine, OPPONENT_PROFILES["balanced"])

        assert not action.is_deploy_action()


class TestBaselineStrength:
    """These are the numbers that decide whether training has headroom."""

    @pytest.mark.parametrize("name", COMPETITIVE)
    def test_competitive_opponents_are_not_walkovers(self, name):
        """Untrained genomes must not simply beat the baseline.

        Before the opponents were rewritten they lost to random genomes 83-100%
        of the time, so fitness hit its ceiling almost immediately.
        """
        wins, draws, losses = _record_against(name)
        total = wins + draws + losses
        assert total > 0
        win_rate = wins / total
        assert win_rate <= 0.75, (
            f"untrained genomes beat '{name}' {win_rate:.0%} of the time; "
            f"this baseline is too weak to train against"
        )

    @pytest.mark.parametrize("name", COMPETITIVE)
    def test_competitive_opponents_are_beatable(self, name):
        """An unwinnable baseline gives no gradient either.

        Measured rates sit around 25-30%; the bound is loose so the test
        tracks "there is headroom in both directions" rather than a
        particular balance point.
        """
        wins, draws, losses = _record_against(name)
        total = wins + draws + losses
        win_rate = wins / total
        assert win_rate >= 0.05, (
            f"untrained genomes beat '{name}' only {win_rate:.0%} of the time; "
            f"this baseline leaves almost no gradient to climb"
        )

    def test_fitness_still_separates_agents(self):
        scores = [
            _run_matches(i, WorkerConfig(seed=31, match_count=2),
                         _genome(400 + i), "balanced", None).fitness
            for i in range(8)
        ]
        assert len(set(scores)) > 1
        assert np.std(scores) > 0.0
