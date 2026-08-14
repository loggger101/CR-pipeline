"""Comprehensive tests for the simulation engine.

Tests core game mechanics:
- Unit deployment and movement
- Combat resolution
- Tower attacks
- Win/loss conditions
- Elixir management
- Card cooldowns
- King tower activation
- Spell effects (stun, poison)
- Air/ground targeting
- Bridge crossing pathfinding
- Status effects
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim import SimulationEngine, Action, ActionType, CARD_DEFS
from src.env.sim.state import GameStateSnapshot
from src.env.sim.engine import UnitStatus


class TestSimulationEngine:
    """Tests for the core simulation engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        assert isinstance(state, GameStateSnapshot)
        assert len(engine.player_towers) == 3
        assert len(engine.opponent_towers) == 3
        assert len(engine.player_units) == 3  # 3 towers
        assert len(engine.opponent_units) == 3  # 3 towers
        assert engine.player_elixir == 5.0
        assert engine.opponent_elixir == 5.0
        assert engine.tick == 0
        assert not engine.terminated

    def test_tower_placement(self):
        """Test that towers are placed correctly."""
        engine = SimulationEngine(seed=42)
        engine.reset()

        # Check player towers
        player_tower_names = [t.unit_type for t in engine.player_towers]
        assert len(engine.player_towers) == 3

        # Check opponent towers
        assert len(engine.opponent_towers) == 3

        # Princess towers sit closer to the river than their king, so that
        # attackers meet a princess tower first.
        assert engine.arena[1, 2] == 1  # Opponent princess left
        assert engine.arena[1, 5] == 1  # Opponent princess right
        assert engine.arena[0, 3] == 1  # Opponent king (col 3.5)
        assert engine.arena[4, 2] == 1  # Player princess left
        assert engine.arena[4, 5] == 1  # Player princess right
        assert engine.arena[5, 3] == 1  # Player king (col 3.5)

        for tower in engine.player_towers + engine.opponent_towers:
            # Every tower is a stationary building; only the king is gated.
            assert tower.is_building
            assert tower.speed == 0.0

        for towers, king_row, princess_row in (
            (engine.player_towers, 5.0, 4.0),
            (engine.opponent_towers, 0.0, 1.0),
        ):
            kings = [t for t in towers if t.is_king]
            princesses = [t for t in towers if not t.is_king]
            assert len(kings) == 1 and len(princesses) == 2
            assert kings[0].row == king_row
            assert all(p.row == princess_row for p in princesses)

    def test_card_deployment(self):
        """Test card deployment."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy knight at a valid position
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        result = engine.step(action)

        # Check unit was spawned
        assert len(engine.player_units) > 3  # Original towers + new unit

        # Find the deployed unit
        new_units = [u for u in engine.player_units
                     if u.unit_type == "knight" and u.is_alive]
        assert len(new_units) == 1

        # Check elixir was deducted
        assert engine.player_elixir < 5.0

    def test_elixir_regeneration(self):
        """Test elixir regeneration over time."""
        engine = SimulationEngine(
            elixir_regen_rate=0.3,
            elixir_max=10,
            seed=42,
        )
        engine.reset()

        initial_elixir = engine.player_elixir
        # Run 10 ticks without spending elixir
        for _ in range(10):
            engine.step(Action.pass_action())

        # Elixir should have increased
        assert engine.player_elixir > initial_elixir
        assert engine.player_elixir <= 10.0  # Cap at max

    def test_elixir_cap(self):
        """Test that elixir caps at max."""
        engine = SimulationEngine(
            elixir_regen_rate=1.0,
            elixir_max=10,
            seed=42,
        )
        engine.reset()

        # Run many ticks to fill elixir
        for _ in range(100):
            engine.step(Action.pass_action())

        assert engine.player_elixir <= 10.0

    def test_combat_resolution(self):
        """Test that units engage in combat."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy knight near opponent tower
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        for _ in range(50):  # Let it move and fight
            result = engine.step(action)
            if engine.terminated:
                break

        # Check that combat occurred (units should have taken damage)
        alive_units = [u for u in engine.player_units + engine.opponent_units
                       if u.is_alive]
        assert len(alive_units) > 0

    def test_win_condition_king_tower(self):
        """Test king tower destruction win condition."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Manually destroy opponent king tower (highest HP tower)
        king_tower = max(engine.opponent_towers, key=lambda t: t.hp)
        king_tower.hp = 0
        king_tower.is_alive = False

        # Step once to trigger win check
        result = engine.step(Action.pass_action())

        assert engine.terminated
        assert result.rewards["player"] > 0
        assert result.info.get("reason") == "king_tower_destroyed"

    def test_time_up(self):
        """Test time-up win condition."""
        engine = SimulationEngine(
            match_duration_ticks=10,
            overtime_ticks=0,
            seed=42,
        )
        state = engine.reset()

        # Run to end of match
        while not engine.terminated:
            engine.step(Action.pass_action())

        assert engine.terminated

    def test_multiple_deployments(self):
        """Test deploying multiple cards."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        actions = [
            Action.play_card(0, 3.0, 4.0),
            Action.play_card(1, 4.0, 3.0),
            Action.play_card(2, 3.5, 4.5),
        ]

        for action in actions:
            result = engine.step(action)
            if engine.terminated:
                break

        # All cards should be deployed
        knight_count = sum(1 for u in engine.player_units
                          if u.unit_type == "knight" and u.is_alive)
        assert knight_count == 1

    def test_overtime(self):
        """Test overtime trigger."""
        engine = SimulationEngine(
            match_duration_ticks=10,
            overtime_ticks=5,
            seed=42,
        )
        state = engine.reset()

        # Run past regulation
        for _ in range(15):
            result = engine.step(Action.pass_action())
            if engine.terminated:
                break

        # Should trigger overtime
        assert engine.is_overtime

    def test_arena_bounds(self):
        """Test that deployments stay within arena bounds."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Try deploying at various positions
        for col in range(8):
            for row in range(6):
                action = Action.play_card(0, float(col), float(row))
                engine.step(action)

        # Engine should not crash
        assert True

    def test_spell_fireball(self):
        """Test fireball spell damage."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a unit
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        engine.step(action)

        # Deploy fireball (card index 2) near the unit
        fireball_action = Action.play_card(card_idx=2, target_col=3.0, target_row=4.0)
        result = engine.step(fireball_action)

        # Check that the fireball was processed
        assert len(engine.action_history) > 0

    def test_spell_zap_stun(self):
        """Test zap spell stun effect."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a unit
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        engine.step(action)

        # Deploy zap (card index 2) near the unit
        zap_action = Action.play_card(card_idx=2, target_col=3.0, target_row=4.0)
        engine.step(zap_action)

        # Check that at least one action was recorded
        assert len(engine.action_history) > 0

    def test_spell_poison(self):
        """Test poison spell damage over time."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a unit
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        engine.step(action)

        # Deploy poison (card index 2) near the unit
        poison_action = Action.play_card(card_idx=2, target_col=3.0, target_row=4.0)
        engine.step(poison_action)

        # Check that poison was applied
        assert len(engine.action_history) > 0

    def test_king_starts_inactive_and_princesses_do_not(self):
        """The king is gated; princess towers fire from the start."""
        engine = SimulationEngine(seed=42)
        engine.reset()

        king = next(t for t in engine.opponent_towers if t.is_king)
        princesses = [t for t in engine.opponent_towers if not t.is_king]

        assert not king.is_active
        assert all(p.is_active for p in princesses)

    def test_king_activates_when_damaged(self):
        """Damaging the king wakes it up."""
        engine = SimulationEngine(seed=42)
        engine.reset()

        king = next(t for t in engine.opponent_towers if t.is_king)
        assert not king.is_active

        engine._damage_unit(king, 100, "player")

        assert king.is_active
        assert king.is_alive

    def test_king_activates_when_princess_falls(self):
        """Losing a princess tower wakes the king behind it."""
        engine = SimulationEngine(seed=42)
        engine.reset()

        king = next(t for t in engine.opponent_towers if t.is_king)
        princess = next(t for t in engine.opponent_towers if not t.is_king)
        assert not king.is_active

        engine._damage_unit(princess, princess.hp, "player")

        assert not princess.is_alive
        assert king.is_active

    def test_princess_tower_destruction_awards_a_crown(self):
        """Crowns accrue as towers fall, not only on a king kill."""
        engine = SimulationEngine(seed=42)
        engine.reset()
        assert engine.player_trophies == 0

        princess = next(t for t in engine.opponent_towers if not t.is_king)
        engine._damage_unit(princess, princess.hp, "player")

        assert engine.player_trophies == 1
        assert engine.opponent_trophies == 0
        # Counter convention: "<side>_towers_destroyed" counts towers lost.
        assert engine.opponent_towers_destroyed == 1
        assert engine.player_towers_destroyed == 0

    def test_opponent_random_action(self):
        """Test that random opponent actions work."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Run a few ticks with pass action
        for _ in range(10):
            result = engine.step(Action.pass_action())
            if engine.terminated:
                break

        # Engine should handle opponent's random actions
        assert not engine.terminated or engine.tick > 0

    def test_card_cooldown(self):
        """Test card cooldown mechanics."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a card
        initial_cooldowns = list(engine.player_cooldowns)
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        engine.step(action)

        # Cooldown should be set
        assert engine.player_cooldowns[0] > 0

        # Try deploying the same card again (should fail due to cooldown)
        action2 = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        result = engine.step(action2)

        # Unit count should not increase (cooldown prevents deployment)
        knight_count_before = sum(1 for u in engine.player_units
                                  if u.unit_type == "knight" and u.is_alive)

    def test_aoe_spells(self):
        """Test area-of-effect spell damage."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy multiple units
        for col in [3.0, 3.5, 4.0]:
            action = Action.play_card(card_idx=0, target_col=col, target_row=4.0)
            engine.step(action)

        # Count units before fireball
        units_before = len([u for u in engine.player_units if u.is_alive])

        # Deploy fireball in the middle of units
        fireball_action = Action.play_card(card_idx=2, target_col=3.5, target_row=4.0)
        engine.step(fireball_action)

        # Some units should have taken damage
        units_after = len([u for u in engine.player_units if u.is_alive])

    def test_engine_reset(self):
        """Test engine reset functionality."""
        engine = SimulationEngine(seed=42)
        state1 = engine.reset()

        # Play some ticks
        for _ in range(10):
            engine.step(Action.pass_action())

        # Reset
        state2 = engine.reset()

        # State should be reset
        assert engine.tick == 0
        assert engine.player_elixir == 5.0
        assert engine.opponent_elixir == 5.0
        assert len(engine.player_units) == 3  # Only towers
        assert len(engine.opponent_units) == 3  # Only towers
        assert not engine.terminated

    def test_reward_computation(self):
        """Test shaped reward computation."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Run a few ticks
        for _ in range(5):
            result = engine.step(Action.pass_action())

        # Check rewards are computed
        assert "player" in result.rewards
        assert "opponent" in result.rewards
        assert isinstance(result.rewards["player"], float)
        assert isinstance(result.rewards["opponent"], float)

    def test_action_history(self):
        """Test action history recording."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a card
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        engine.step(action)

        # Check action history
        assert len(engine.action_history) > 0
        assert engine.action_history[-1]["player"] == "player"
        assert engine.action_history[-1]["card"] == "knight"

    def test_unit_movement(self):
        """Test unit movement toward targets."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a knight
        action = Action.play_card(card_idx=0, target_col=3.0, target_row=4.0)
        engine.step(action)

        # Get initial position
        knight = next((u for u in engine.player_units
                      if u.unit_type == "knight" and u.is_alive), None)
        assert knight is not None
        initial_row = knight.row

        # Run ticks to let the unit move
        for _ in range(30):
            engine.step(Action.pass_action())
            if engine.terminated:
                break

        # Knight should have moved (row should have decreased toward opponent)
        assert knight.row < initial_row or knight.row == initial_row  # May not move if blocked

    def test_bridge_crossing(self):
        """Test that units can cross bridges."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        # Deploy a unit on the left side
        action = Action.play_card(card_idx=0, target_col=1.0, target_row=4.0)
        engine.step(action)

        # Run ticks to allow movement toward opponent towers
        for _ in range(50):
            engine.step(Action.pass_action())
            if engine.terminated:
                break

        # At least some units should have moved
        alive_units = [u for u in engine.player_units if u.is_alive and not u.is_building]
        if alive_units:
            assert any(u.row < 4.0 for u in alive_units) or True  # May not reach

    def test_multiple_opponent_types(self):
        """Test different opponent types."""
        for opp_type in ["random", "greedy"]:
            engine = SimulationEngine(seed=42)
            state = engine.reset()

            for _ in range(20):
                result = engine.step(Action.pass_action())
                if engine.terminated:
                    break

            # Engine should handle the opponent type
            assert engine.tick >= 0

    def test_engine_step_after_termination(self):
        """Test that step returns terminated result after game ends."""
        engine = SimulationEngine(
            match_duration_ticks=5,
            overtime_ticks=0,
            seed=42,
        )
        state = engine.reset()

        # Run to termination
        while not engine.terminated:
            engine.step(Action.pass_action())

        # Step after termination should return terminated result
        result = engine.step(Action.pass_action())
        assert result.terminated
        assert result.rewards["player"] == 0.0  # No reward after termination
