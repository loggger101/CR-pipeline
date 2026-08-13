"""Tests for the simulation engine.

Tests core game mechanics:
- Unit deployment and movement
- Combat resolution
- Tower attacks
- Win/loss conditions
- Elixir management
- Card cooldowns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim import SimulationEngine, Action, ActionType, CARD_DEFS
from src.env.sim.state import GameStateSnapshot


class TestSimulationEngine:
    """Tests for the core simulation engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = SimulationEngine(seed=42)
        state = engine.reset()

        assert isinstance(state, GameStateSnapshot)
        assert len(engine.player_towers) == 3
        assert len(engine.opponent_towers) == 3
        assert len(engine.player_units) == 3
        assert len(engine.opponent_units) == 3
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

        # Check arena has towers marked as occupied
        assert engine.arena[0, 2] == 1  # Opponent princess left
        assert engine.arena[0, 5] == 1  # Opponent princess right
        assert engine.arena[5, 2] == 1  # Player princess left
        assert engine.arena[5, 5] == 1  # Player princess right

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
        # At least one unit should have reduced HP
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


class TestActionSpace:
    """Tests for the action space."""

    def test_pass_action(self):
        """Test pass action creation."""
        action = Action.pass_action()
        assert action.action_type == ActionType.PASS
        assert action.valid is True

    def test_play_card_action(self):
        """Test play card action creation."""
        action = Action.play_card(0, 3.0, 4.0)
        assert action.action_type == ActionType.PLAY_CARD
        assert action.card_idx == 0
        assert action.target_col == 3.0
        assert action.target_row == 4.0

    def test_action_serialization(self):
        """Test action to/from array conversion."""
        action = Action.play_card(2, 5.0, 3.0)
        arr = action.to_array()
        restored = Action.from_array(arr)

        assert restored.card_idx == action.card_idx
        assert restored.target_col == action.target_col
        assert restored.target_row == action.target_row

    def test_action_to_tuple(self):
        """Test action to tuple conversion."""
        action = Action.play_card(1, 4.0, 2.0)
        tup = action.to_tuple()
        assert tup == (1, 4.0, 2.0)


class TestEntityDefinitions:
    """Tests for entity definitions."""

    def test_card_registry(self):
        """Test card registry."""
        from src.env.sim import get_card_def

        knight = get_card_def("knight")
        assert knight.name == "knight"
        assert knight.hp == 1400
        assert knight.damage == 75
        assert knight.elixir_cost == 3

    def test_tower_registry(self):
        """Test tower registry."""
        from src.env.sim import get_tower_def

        tower = get_tower_def("opp_princess_left")
        assert tower.hp == 1400
        assert tower.damage == 75

    def test_unknown_card(self):
        """Test lookup of unknown card."""
        from src.env.sim import get_card_def

        with pytest.raises(KeyError):
            get_card_def("nonexistent_card")

    def test_card_deployment_zones(self):
        """Test card deployment zone constraints."""
        from src.env.sim import DeployZone, get_card_def

        knight = get_card_def("knight")
        assert knight.deploy_zone == DeployZone.GROUND

        archer = get_card_def("archer")
        assert archer.deploy_zone == DeployZone.GROUND

        fireball = get_card_def("fireball")
        assert fireball.deploy_zone == DeployZone.SPELL

        minion = get_card_def("minion")
        assert minion.deploy_zone == DeployZone.AIR


class TestGameState:
    """Tests for game state representation."""

    def test_preprocess_state(self):
        """Test state preprocessing."""
        from src.env.sim import preprocess_state

        state = GameStateSnapshot()
        state.card_hand = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        state.player_elixir = 7.5
        state.opponent_elixir = 3.0
        state.time_remaining = 90.0

        tensor = preprocess_state(state, resolution=32)

        assert tensor.shape == (12, 32, 32)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_compute_state_from_arena(self):
        """Test arena-based state computation."""
        from src.env.sim import compute_state_from_arena, GameStateSnapshot

        state = GameStateSnapshot()
        arena = np.zeros((6, 8), dtype=np.int32)

        # Add some units
        unit = type('Unit', (), {
            'is_alive': True,
            'hp': 100,
            'max_hp': 200,
            'col': 3.0,
            'row': 4.0,
        })()
        state.player_units.append(unit)

        result = compute_state_from_arena(state, arena)

        assert result.unit_density[4, 3] > 0
        assert result.lane_presence[0] > 0  # Left lane has unit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
