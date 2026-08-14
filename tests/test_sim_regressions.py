"""Regression tests for simulation-engine defects.

Each class here pins down behaviour that was previously wrong in a way the
existing suite did not detect. Grouped separately from ``test_sim_engine.py``
so the intent -- "this specific thing was broken" -- stays legible.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim import Action, ActionType, CARD_DEFS, SimulationEngine
from src.env.sim.engine import UnitStatus


def _engine(**kwargs) -> SimulationEngine:
    kwargs.setdefault("seed", 3)
    kwargs.setdefault("record_replay", False)
    engine = SimulationEngine(**kwargs)
    engine.reset()
    return engine


class TestCardCycling:
    """The hand rotates on play, not on a timer.

    The engine previously replaced all four hand slots every third tick
    regardless of what was played, so a hand index carried no stable meaning
    and a policy could not learn one.
    """

    def test_hand_is_stable_when_nothing_is_played(self):
        engine = _engine()
        original = list(engine.player_hand)

        for _ in range(30):
            engine.step(Action.pass_action())

        assert engine.player_hand == original

    def test_playing_a_card_replaces_only_that_slot(self):
        engine = _engine()
        engine.player_elixir = 10.0
        before = list(engine.player_hand)

        engine.step(Action.play_card(card_idx=1, target_col=3.0, target_row=4.0))

        assert engine.player_hand[1] != before[1]
        assert engine.player_hand[0] == before[0]
        assert engine.player_hand[2] == before[2]
        assert engine.player_hand[3] == before[3]

    def test_played_card_returns_to_the_back_of_the_deck(self):
        engine = _engine()
        engine.player_elixir = 10.0
        played = engine.player_hand[0]

        engine.step(Action.play_card(card_idx=0, target_col=3.0, target_row=4.0))

        assert engine.player_deck_queue[-1] == played

    def test_rotation_preserves_the_card_multiset(self):
        engine = _engine()
        expected = sorted(engine.player_hand + engine.player_deck_queue)

        for _ in range(40):
            engine.player_elixir = 10.0
            engine.step(Action.play_card(card_idx=0, target_col=3.0, target_row=4.0))

        assert sorted(engine.player_hand + engine.player_deck_queue) == expected


class TestDeploymentZones:
    """Units are confined to their own half; spells are not."""

    def test_unit_cannot_be_deployed_on_the_enemy_half(self):
        engine = _engine()
        engine.player_elixir = 10.0
        knight_idx = engine.player_hand.index("knight")

        engine.step(Action.play_card(knight_idx, target_col=3.0, target_row=1.0))

        assert not any(u.unit_type == "knight" for u in engine.player_units)
        assert engine.action_history == []

    def test_spell_may_target_the_enemy_half(self):
        """Damage spells were locked to the caster's own side, which left them
        unable to reach anything worth hitting."""
        engine = _engine()
        engine.player_elixir = 10.0
        fireball_idx = engine.player_hand.index("fireball")
        tower = next(t for t in engine.opponent_towers if not t.is_king)
        hp_before = tower.hp

        engine.step(Action(action_type=ActionType.DEPLOY_SPELL,
                           card_index=fireball_idx,
                           target_col=tower.col, target_row=tower.row))

        assert tower.hp < hp_before

    def test_spell_kill_awards_a_crown(self):
        """Spell damage routes through the same death handling as combat."""
        engine = _engine()
        engine.player_elixir = 10.0
        tower = next(t for t in engine.opponent_towers if not t.is_king)
        tower.hp = 1.0
        fireball_idx = engine.player_hand.index("fireball")

        engine.step(Action(action_type=ActionType.DEPLOY_SPELL,
                           card_index=fireball_idx,
                           target_col=tower.col, target_row=tower.row))

        assert not tower.is_alive
        assert engine.player_trophies == 1
        assert engine.opponent_towers_destroyed == 1

    def test_spell_damage_is_booked_as_actual_damage(self):
        """Damage booked must be what was dealt, not the victim's max HP."""
        engine = _engine()
        tower = next(t for t in engine.opponent_towers if not t.is_king)
        # Radius small enough to catch only this tower.
        engine._apply_spell_damage(tower.col, tower.row, 200.0, 0.3,
                                   "player", "fireball")
        assert engine._tower_damage_dealt["opponent"] == pytest.approx(200.0)
        assert tower.hp == pytest.approx(tower.max_hp - 200.0)


class TestStatusEffects:
    """Status effects must actually modify the unit.

    The engine passed ``UnitStatus`` members into a method that compared
    against bare ints, so every speed modifier silently no-opped.
    """

    def test_stun_zeroes_movement_speed(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        unit = engine.player_units[-1]
        assert unit.speed > 0

        unit.apply_status(UnitStatus.STUNNED, 10, "zap")

        assert unit.speed == 0.0

    def test_slow_halves_speed_and_restores_it(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        unit = engine.player_units[-1]
        base = unit.speed

        unit.apply_status(UnitStatus.SLOWED, 2, "tornado")
        assert unit.speed == pytest.approx(base * 0.5)

        unit.clear_status()
        assert unit.speed == pytest.approx(base)

    def test_status_expires_after_its_duration(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        unit = engine.player_units[-1]
        base = unit.speed
        unit.apply_status(UnitStatus.STUNNED, 3, "zap")

        for _ in range(3):
            unit.update_status()

        assert unit.status == UnitStatus.NONE
        assert unit.speed == pytest.approx(base)

    def test_effects_do_not_compound(self):
        """Re-applying a slow must not repeatedly halve the base speed."""
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        unit = engine.player_units[-1]
        base = unit.speed

        for _ in range(5):
            unit.apply_status(UnitStatus.SLOWED, 10, "tornado")

        assert unit.speed == pytest.approx(base * 0.5)

    def test_towers_cannot_be_stunned(self):
        engine = _engine()
        tower = next(t for t in engine.opponent_towers if not t.is_king)
        engine._apply_stun(tower.col, tower.row, 3.0, "player")
        assert tower.status == UnitStatus.NONE


class TestUnitLifecycle:
    """Corpses, death effects, and combat cadence."""

    def test_dead_units_are_reclaimed(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        unit = engine.player_units[-1]
        count_before = len(engine.player_units)

        unit.take_damage(unit.max_hp)
        engine.step(Action.pass_action())

        assert len(engine.player_units) < count_before
        assert all(u.is_alive or u.is_building for u in engine.player_units)

    def test_towers_are_retained_after_death(self):
        """Win conditions and rendering both read the tower lists."""
        engine = _engine()
        tower = next(t for t in engine.player_towers if not t.is_king)

        engine._damage_unit(tower, tower.max_hp, "opponent")
        engine.step(Action.pass_action())

        assert tower in engine.player_units
        assert len(engine.player_towers) == 3

    def test_death_split_runs_exactly_once(self):
        """A Golem splits into exactly two Golem Minis.

        The old nested loop iterated the combined unit list once per side, and
        keyed off card names ("elixir_golem"/"mini_golem") absent from the
        registry, so no card split at all.
        """
        engine = _engine()
        engine._spawn_unit("golem", 3.0, 4.0, "player")
        golem = engine.player_units[-1]
        assert golem.max_hp == CARD_DEFS["golem"].hitpoints

        golem.take_damage(golem.max_hp)
        engine.step(Action.pass_action())

        minis = [u for u in engine.player_units if u.unit_type == "golem_mini"]
        assert len(minis) == 2

    def test_a_tank_deploys_as_itself_not_as_its_death_spawn(self):
        """Lava Hound must arrive as a Lava Hound, not as a Lava Pup."""
        engine = _engine()
        engine._spawn_unit("lava_hound", 3.0, 4.0, "player")
        hound = engine.player_units[-1]
        assert hound.unit_type == "lava_hound"
        assert hound.max_hp == CARD_DEFS["lava_hound"].hitpoints

    def test_swarm_cards_still_deploy_multiple_units(self):
        engine = _engine()
        engine.player_elixir = 10.0
        before = len(engine.player_units)
        engine._deploy_card("minions", 3.0, 4.0, "player", 0)
        spawned = len(engine.player_units) - before
        assert spawned == CARD_DEFS["minions"].spawn_count
        assert all(u.unit_type == "minion"
                   for u in engine.player_units[before:])

    def test_take_damage_flags_death_once(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        unit = engine.player_units[-1]

        unit.take_damage(unit.max_hp)
        assert unit.just_died
        assert unit.take_damage(50) == 0.0  # already dead

    def test_attack_cadence_follows_attack_speed(self):
        """attack_speed is in seconds and scales by the engine timebase; the
        old `60 * attack_speed` left ~72 ticks between swings."""
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        knight = engine.player_units[-1]

        expected = round(knight.attack_speed * engine.TICKS_PER_SECOND)
        assert engine._attack_cooldown_ticks(knight) == expected
        assert 1 <= expected <= 30

    def test_king_tower_does_not_move(self):
        """The king was not flagged as a building, so it walked up the arena."""
        engine = _engine()
        king = next(t for t in engine.player_towers if t.is_king)
        start = (king.col, king.row)

        for _ in range(60):
            if engine.terminated:
                break
            engine.step(Action.pass_action())

        assert (king.col, king.row) == start


class TestTargeting:
    """Target acquisition respects range and air/ground rules."""

    def test_unit_already_in_range_does_not_advance(self):
        """The movement step used to close to 0.1 tiles regardless of range."""
        engine = _engine()
        engine._spawn_unit("musketeer", 3.5, 4.0, "player")
        musketeer = engine.player_units[-1]
        target = engine._find_target(musketeer)
        assert target is not None
        assert engine._distance(musketeer, target) <= musketeer.range

        start = (musketeer.col, musketeer.row)
        engine._move_units()

        assert (musketeer.col, musketeer.row) == start

    def test_melee_unit_stops_at_contact_not_on_top_of_its_target(self):
        engine = _engine()
        engine._spawn_unit("knight", 7.0, 5.0, "player")
        knight = engine.player_units[-1]
        assert knight.range < 1.5

        for _ in range(600):
            if engine.terminated or not knight.is_alive:
                break
            engine.step(Action.pass_action())
            target = engine._find_target(knight)
            if target is not None and engine._distance(knight, target) <= knight.range:
                break

        target = engine._find_target(knight)
        if knight.is_alive and target is not None:
            # Approach halts at weapon range rather than overlapping.
            assert engine._distance(knight, target) > 0.1

    def test_ground_only_unit_ignores_air_targets(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        knight = engine.player_units[-1]
        knight.can_target_air = False

        engine._spawn_unit("minion", 3.0, 3.8, "opponent")
        air = [u for u in engine.opponent_units if u.is_air]
        assert air, "expected an air unit to be spawned"

        assert not engine._can_attack(knight, air[0])
        assert engine._can_attack(knight, knight)  # ground is still valid

    def test_air_unit_can_still_hit_ground(self):
        """Air units were forced to can_target_ground=False, so Minions and
        Baby Dragon could not attack anything on the floor."""
        engine = _engine()
        engine._spawn_unit("minion", 3.0, 4.0, "player")
        minion = engine.player_units[-1]
        assert minion.is_air
        assert minion.can_target_ground == bool(CARD_DEFS["minion"].target_ground)
        assert minion.can_target_ground

    def test_building_targeter_prefers_towers(self):
        engine = _engine()
        engine._spawn_unit("giant", 3.0, 4.0, "player")
        giant = engine.player_units[-1]
        engine._spawn_unit("knight", 3.0, 3.5, "opponent")

        target = engine._find_target(giant)

        assert target is not None
        assert target.is_building


class TestMatchOutcomes:
    """Crowns, overtime, and end-of-match scoring."""

    def test_crowns_decide_a_timed_match(self):
        engine = _engine(match_duration_ticks=20, overtime_ticks=0)
        tower = next(t for t in engine.opponent_towers if not t.is_king)
        engine._damage_unit(tower, tower.max_hp, "player")

        result = None
        while not engine.terminated:
            result = engine.step(Action.pass_action())

        assert result.info["winner"] == "player"
        assert result.info["reason"] == "time_up"
        assert result.rewards["player"] > 0

    def test_level_match_is_a_draw(self):
        engine = _engine(match_duration_ticks=20, overtime_ticks=0)

        result = None
        while not engine.terminated:
            result = engine.step(Action.pass_action())

        assert result.info["winner"] == "tie"

    def test_overtime_does_not_report_truncation(self):
        """Entering overtime is not the end of the episode."""
        engine = _engine(match_duration_ticks=10, overtime_ticks=20)

        for _ in range(12):
            result = engine.step(Action.pass_action())
            if engine.is_overtime:
                assert not result.terminated
                assert not result.truncated
                return
        pytest.fail("overtime never triggered")

    def test_a_crown_lead_ends_regulation_without_overtime(self):
        engine = _engine(match_duration_ticks=20, overtime_ticks=100)
        tower = next(t for t in engine.opponent_towers if not t.is_king)
        engine._damage_unit(tower, tower.max_hp, "player")

        while not engine.terminated:
            engine.step(Action.pass_action())

        assert not engine.is_overtime
        assert engine.tick <= 25

    def test_reset_with_a_seed_is_reproducible(self):
        """Per-match seeding is what makes N matches informative."""
        engine = SimulationEngine(seed=1, record_replay=False)
        engine.reset(seed=100)
        first = list(engine.opponent_hand)
        engine.reset(seed=999)
        engine.reset(seed=100)
        assert list(engine.opponent_hand) == first


class TestEconomy:
    """Elixir and timebase calibration."""

    def test_default_elixir_rate_matches_the_real_game(self):
        engine = _engine()
        # 1 elixir per 2.8 seconds at 10 ticks/second.
        assert engine.elixir_regen_rate == pytest.approx(1.0 / 2.8 / 10)

    def test_explicit_elixir_rate_is_respected(self):
        engine = _engine(elixir_regen_rate=0.5)
        assert engine.elixir_regen_rate == 0.5

    def test_regulation_is_three_minutes_of_game_time(self):
        engine = _engine()
        assert engine.match_duration_ticks / engine.TICKS_PER_SECOND == 180.0

    def test_move_speed_is_converted_to_per_tick(self):
        engine = _engine()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        knight = engine.player_units[-1]
        expected = CARD_DEFS["knight"].move_speed / engine.TICKS_PER_SECOND
        assert knight.speed == pytest.approx(expected)
        assert knight.base_speed == pytest.approx(expected)
