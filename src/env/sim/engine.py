"""Core simulation engine for Clash Royale.

Tick-based game loop that processes:
1. Elixir regeneration
2. Unit movement toward targets
3. Unit-to-unit and unit-to-tower combat
4. Spell effects
5. Tower attacks
6. Win/loss detection

Designed for speed: each tick processes ~60+ game events in pure NumPy.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .entities import (
    CARD_DEFS, TOWER_DEFS, CardDefinition, DeployZone, EntityType,
    TargetPreference, TowerDefinition, UnitDefinition,
)
from .actions import Action, ActionType
from .state import GameStateSnapshot, UnitState, compute_state_from_arena


@dataclass
class SimulationStepResult:
    """Result of a single simulation step.

    Attributes:
        rewards: Dict of reward signals for each player.
        terminated: Whether the game has ended.
        truncated: Whether the game was truncated (e.g., timeout).
        info: Additional diagnostic information.
    """
    rewards: Dict[str, float] = field(default_factory=lambda: {"player": 0.0, "opponent": 0.0})
    terminated: bool = False
    truncated: bool = False
    info: Dict = field(default_factory=dict)


class SimulationEngine:
    """Clash Royale simulation engine.

    Manages the game state, processes actions, and runs the tick-based
    game loop. Each instance represents a single match.

    Arena layout (8 columns x 6 rows):
        Row 0-1: Opponent territory (opponent towers)
        Row 2:   Bridge area (left bridge ~col 3, right bridge ~col 4)
        Row 3:   Mid-field
        Row 4-5: Player territory (player towers)

    Towers are placed at:
        Opponent princess left:  (col 2, row 0)
        Opponent princess right: (col 5, row 0)
        Opponent king:           (col 3.5, row 0)
        Player princess left:    (col 2, row 5)
        Player princess right:   (col 5, row 5)
        Player king:             (col 3.5, row 5)
    """

    # Arena dimensions
    GRID_COLS = 8
    GRID_ROWS = 6

    # Bridge columns
    BRIDGE_COLS = [3, 4]

    # Default deck for training
    DEFAULT_DECK = [
        "knight", "archers", "fireball", "musketeer",
        "mini_pekka", "valkyrie", "baby_dragon", "wizard",
    ]

    def __init__(
        self,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
        match_duration_ticks: int = 1800,
        overtime_ticks: int = 120,
        elixir_regen_rate: float = 0.3,
        elixir_max: int = 10,
        double_elixir_overtime: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the simulation engine.

        Args:
            deck: Player's card deck (8 cards). Defaults to DEFAULT_DECK.
            opponent_deck: Opponent's card deck. Defaults to a shuffled DEFAULT_DECK.
            match_duration_ticks: Length of regulation match.
            overtime_ticks: Additional ticks for overtime.
            elixir_regen_rate: Elixir regenerated per tick.
            elixir_max: Maximum elixir capacity.
            double_elixir_overtime: Whether elixir doubles in overtime.
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.match_duration_ticks = match_duration_ticks
        self.overtime_ticks = overtime_ticks
        self.elixir_regen_rate = elixir_regen_rate
        self.elixir_max = elixir_max
        self.double_elixir_overtime = double_elixir_overtime

        # Set up decks
        if deck is not None:
            self.player_deck = list(deck)
        else:
            self.player_deck = list(self.DEFAULT_DECK)
        if opponent_deck is not None:
            self.opponent_deck = list(opponent_deck)
        else:
            self.opponent_deck = list(self.DEFAULT_DECK)
            self.rng.shuffle(self.opponent_deck)

        # Current card hands (4 cards each)
        self.player_hand: List[str] = []
        self.opponent_hand: List[str] = []
        self.player_deck_queue: List[str] = []
        self.opponent_deck_queue: List[str] = []

        # Cooldowns per hand slot
        self.player_cooldowns: np.ndarray = np.zeros(4, dtype=np.int32)
        self.opponent_cooldowns: np.ndarray = np.zeros(4, dtype=np.int32)

        # Elixir levels
        self.player_elixir: float = 5.0
        self.opponent_elixir: float = 5.0

        # Game state
        self.tick: int = 0
        self.is_overtime: bool = False
        self.terminated: bool = False
        self.truncated: bool = False

        # Unit tracking
        self.player_units: List[UnitState] = []
        self.opponent_units: List[UnitState] = []
        self.player_towers: List[UnitState] = []
        self.opponent_towers: List[UnitState] = []

        # Trophy tracking
        self.player_trophies: int = 0
        self.opponent_trophies: int = 0

        # Towers destroyed
        self.player_towers_destroyed: int = 0
        self.opponent_towers_destroyed: int = 0

        # Arena grid: 0 = empty, > 0 = occupied (for placement validation)
        self.arena: np.ndarray = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int32)

        # Action history for replay
        self.action_history: List[dict] = []

        # Initial state for reset
        self._initial_deck = list(self.player_deck)
        self._initial_opponent_deck = list(self.opponent_deck)
        self._initial_seed = seed

        self.reset()

    def reset(self) -> GameStateSnapshot:
        """Reset to initial state and return the first observation.

        Returns:
            Initial GameStateSnapshot.
        """
        self.rng = np.random.RandomState(self._initial_seed)
        self.player_deck = list(self._initial_deck)
        self.opponent_deck = list(self._initial_opponent_deck)
        self.player_deck_queue = list(self.player_deck)
        self.opponent_deck_queue = list(self.opponent_deck)

        # Fill initial hands
        self.player_hand = [self.player_deck_queue.pop(0) for _ in range(4)]
        self.opponent_hand = [self.opponent_deck_queue.pop(0) for _ in range(4)]

        self.player_cooldowns = np.zeros(4, dtype=np.int32)
        self.opponent_cooldowns = np.zeros(4, dtype=np.int32)

        self.player_elixir = 5.0
        self.opponent_elixir = 5.0

        self.tick = 0
        self.is_overtime = False
        self.terminated = False
        self.truncated = False

        self.player_units = []
        self.opponent_units = []
        self.player_towers = []
        self.opponent_towers = []
        self.arena = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int32)
        self.action_history = []

        self.player_trophies = 0
        self.opponent_trophies = 0
        self.player_towers_destroyed = 0
        self.opponent_towers_destroyed = 0

        # Place towers
        self._place_towers()

        return self._get_state()

    def _place_towers(self) -> None:
        """Place all towers on the arena."""
        # Opponent towers
        opp_tower_positions = [
            ("opp_princess_left", 2.0, 0.0),
            ("opp_princess_right", 5.0, 0.0),
            ("opp_king", 3.5, 0.0),
        ]
        for name, col, row in opp_tower_positions:
            defn = TOWER_DEFS[name]
            unit = UnitState(
                unit_type="tower",
                owner="opponent",
                hp=defn.hp,
                max_hp=defn.hp,
                col=col,
                row=row,
                is_alive=True,
                is_building=True,
            )
            self.opponent_towers.append(unit)
            self.opponent_units.append(unit)
            self.arena[int(row), int(col)] = 1  # Occupied

        # Player towers
        player_tower_positions = [
            ("player_princess_left", 2.0, 5.0),
            ("player_princess_right", 5.0, 5.0),
            ("player_king", 3.5, 5.0),
        ]
        for name, col, row in player_tower_positions:
            defn = TOWER_DEFS[name]
            unit = UnitState(
                unit_type="tower",
                owner="player",
                hp=defn.hp,
                max_hp=defn.hp,
                col=col,
                row=row,
                is_alive=True,
                is_building=True,
            )
            self.player_towers.append(unit)
            self.player_units.append(unit)
            self.arena[int(row), int(col)] = 1

    def _get_state(self) -> GameStateSnapshot:
        """Build a GameStateSnapshot from current engine state."""
        # Card hand availability
        card_hand = np.zeros(4, dtype=np.float32)
        for i in range(4):
            if self.player_cooldowns[i] <= 0:
                card_hand[i] = 1.0

        snapshot = GameStateSnapshot(
            card_hand=card_hand,
            player_elixir=self.player_elixir / self.elixir_max,
            opponent_elixir=self.opponent_elixir / self.elixir_max,
            player_units=list(self.player_units),
            opponent_units=list(self.opponent_units),
            player_towers=list(self.player_towers),
            opponent_towers=list(self.opponent_towers),
            player_trophies=self.player_trophies,
            opponent_trophies=self.opponent_trophies,
            player_towers_destroyed=self.player_towers_destroyed,
            opponent_towers_destroyed=self.opponent_towers_destroyed,
            is_overtime=self.is_overtime,
            tick=self.tick,
            max_ticks=self.match_duration_ticks + (self.overtime_ticks if self.is_overtime else 0),
        )

        # Compute arena-based features
        compute_state_from_arena(snapshot, self.arena)

        # Time remaining
        total_ticks = self.match_duration_ticks + (self.overtime_ticks if self.is_overtime else 0)
        snapshot.time_remaining = max(0, total_ticks - self.tick)

        return snapshot

    def step(self, action: Optional[Action] = None,
             opponent_action: Optional[Action] = None) -> SimulationStepResult:
        """Execute one game tick.

        Args:
            action: Player's action this tick.
            opponent_action: Opponent's action this tick.

        Returns:
            SimulationStepResult with rewards and termination status.
        """
        if self.terminated or self.truncated:
            return SimulationStepResult(
                rewards={"player": 0.0, "opponent": 0.0},
                terminated=True, truncated=True,
            )

        # 1. Process player action
        if action is not None:
            self._process_action(action, "player")

        # 2. Process opponent action
        if opponent_action is not None:
            self._process_action(opponent_action, "opponent")

        # 3. Elixir regeneration
        self._regenerate_elixir()

        # 4. Unit movement
        self._move_units()

        # 5. Combat resolution
        self._resolve_combat()

        # 6. Tower attacks
        self._tower_attacks()

        # 7. Check win conditions
        result = self._check_win_conditions()

        # 8. Tick counter
        self.tick += 1

        # 9. Card cycling (every 3 ticks, cycle hand)
        if self.tick % 3 == 0 and not self.terminated:
            self._cycle_hand("player")
            self._cycle_hand("opponent")

        return result

    def _process_action(self, action: Action, player: str) -> None:
        """Process a player's action (card deployment)."""
        if action.action_type != ActionType.PLAY_CARD:
            return

        hand = self.player_hand if player == "player" else self.opponent_hand
        cooldowns = self.player_cooldowns if player == "player" else self.opponent_cooldowns
        elixir = self.player_elixir if player == "player" else self.opponent_elixir

        card_idx = action.card_idx
        if card_idx < 0 or card_idx >= len(hand):
            return

        card_name = hand[card_idx]
        card_def = CARD_DEFS.get(card_name)
        if card_def is None:
            return

        # Check cooldown
        if cooldowns[card_idx] > 0:
            return

        # Check elixir
        if elixir < card_def.elixir_cost:
            return

        # Check deployment zone
        target_col = action.target_col
        target_row = action.target_row

        if player == "player":
            valid_rows = range(3, self.GRID_ROWS)
        else:
            valid_rows = range(0, 3)

        if target_row not in valid_rows or target_col < 0 or target_col >= self.GRID_COLS:
            return

        # Deploy the card
        self._deploy_card(card_name, target_col, target_row, player, card_idx)

        # Deduct elixir and start cooldown
        if player == "player":
            self.player_elixir -= card_def.elixir_cost
            self.player_cooldowns[card_idx] = 10  # 10-tick cooldown
        else:
            self.opponent_elixir -= card_def.elixir_cost
            self.opponent_cooldowns[card_idx] = 10

        # Record action
        self.action_history.append({
            "tick": self.tick,
            "player": player,
            "card": card_name,
            "col": target_col,
            "row": target_row,
        })

    def _deploy_card(self, card_name: str, col: float, row: float,
                     player: str, hand_idx: int) -> None:
        """Deploy a card at the given position."""
        card_def = CARD_DEFS[card_name]

        if card_def.aoe and card_def.hp == 0:
            # Spell: deal damage to all units in range
            self._apply_spell_damage(col, row, card_def.damage, card_def.range,
                                     player)
            return

        if card_def.spawn_count > 0 and card_def.spawn_type:
            # Swarm card: spawn multiple units
            for i in range(card_def.spawn_count):
                offset_x = self.rng.uniform(-0.3, 0.3)
                offset_y = self.rng.uniform(-0.3, 0.3)
                spawn_col = col + offset_x
                spawn_row = row + offset_y
                self._spawn_unit(card_def.spawn_type, spawn_col, spawn_row, player)
        else:
            # Single unit
            self._spawn_unit(card_name, col, row, player)

    def _spawn_unit(self, card_name: str, col: float, row: float,
                    player: str) -> None:
        """Spawn a unit on the arena."""
        card_def = CARD_DEFS[card_name]
        unit = UnitState(
            unit_type=card_name,
            owner=player,
            hp=card_def.hp,
            max_hp=card_def.hp,
            col=col,
            row=row,
            is_alive=True,
            is_building=card_def.is_building,
        )

        if player == "player":
            self.player_units.append(unit)
        else:
            self.opponent_units.append(unit)

    def _apply_spell_damage(self, center_col: float, center_row: float,
                            damage: int, radius: float, caster: str) -> None:
        """Apply spell damage to all units within radius of center."""
        target_units = self.opponent_units if caster == "player" else self.player_units
        for unit in target_units:
            if not unit.is_alive:
                continue
            dist = math.sqrt((unit.col - center_col) ** 2 +
                             (unit.row - center_row) ** 2)
            if dist <= radius:
                unit.take_damage(damage)

    def _regenerate_elixir(self) -> None:
        """Regenerate elixir for both players."""
        regen_rate = self.elixir_regen_rate
        if self.is_overtime and self.double_elixir_overtime:
            regen_rate *= 2

        self.player_elixir = min(self.elixir_max,
                                 self.player_elixir + regen_rate)
        self.opponent_elixir = min(self.elixir_max,
                                   self.opponent_elixir + regen_rate)

    def _move_units(self) -> None:
        """Move all units toward their targets."""
        all_units = self.player_units + self.opponent_units
        for unit in all_units:
            if not unit.is_alive:
                continue

            # Find target
            target = self._find_target(unit)
            if target is None:
                continue

            # Move toward target
            dx = target.col - unit.col
            dy = target.row - unit.row
            dist = math.sqrt(dx ** 2 + dy ** 2)

            if dist <= 0.1:
                continue

            # Check if unit is at bridge and needs to cross
            if unit.col < 3.5 and target.col >= 3.5:
                # Check left bridge
                if not self._bridge_open(3, unit.row):
                    continue
            elif unit.col > 4.5 and target.col <= 4.5:
                # Check right bridge
                if not self._bridge_open(4, unit.row):
                    continue
            elif target.col < 3.5 and unit.col >= 3.5:
                if not self._bridge_open(3, unit.row):
                    continue
            elif target.col > 4.5 and unit.col <= 4.5:
                if not self._bridge_open(4, unit.row):
                    continue

            # Move in direction of target
            move_x = (dx / dist) * min(unit.speed, dist)
            move_y = (dy / dist) * min(unit.speed, dist)

            unit.col += move_x
            unit.row += move_y

            # Clamp to arena
            unit.col = max(0, min(self.GRID_COLS - 1, unit.col))
            unit.row = max(0, min(self.GRID_ROWS - 1, unit.row))

            # Check if reached target
            new_dist = math.sqrt((unit.col - target.col) ** 2 +
                                 (unit.row - target.row) ** 2)
            if new_dist < dist:
                unit.target_col = target.col
                unit.target_row = target.row

    def _bridge_open(self, col: int, unit_row: float) -> bool:
        """Check if a bridge is intact (not destroyed by spell)."""
        # Simplified: bridges are always open in this simulation
        return True

    def _find_target(self, unit: UnitState) -> Optional[UnitState]:
        """Find the best attack target for a unit.

        Priority: towers > units, by proximity.
        """
        enemy_towers = (self.opponent_towers if unit.owner == "player"
                        else self.player_towers)
        enemy_units = (self.opponent_units if unit.owner == "player"
                       else self.player_units)

        # Filter out dead towers
        active_towers = [t for t in enemy_towers if t.is_alive]
        active_units = [u for u in enemy_units if u.is_alive]

        # King tower activation check
        if active_towers:
            king_tower = next((t for t in active_towers if t.is_building and t.unit_type == "tower"),
                              None)
            if king_tower and king_tower.activation_range > 0:
                dist_to_king = math.sqrt((king_tower.col - unit.col) ** 2 +
                                         (king_tower.row - unit.row) ** 2)
                if dist_to_king > king_tower.activation_range:
                    # Remove king tower from targets
                    active_towers = [t for t in active_towers
                                     if not (t.is_building and t.unit_type == "tower")]

        # Check if any enemy is in range
        for tower in active_towers:
            dist = math.sqrt((tower.col - unit.col) ** 2 +
                             (tower.row - unit.row) ** 2)
            if dist <= unit.range:
                return tower

        # No enemy in range, move toward nearest enemy/tower
        candidates = active_towers + active_units
        if not candidates:
            return None

        # Prefer towers
        if active_towers:
            candidates = active_towers

        # Find nearest
        best = min(candidates,
                   key=lambda t: math.sqrt((t.col - unit.col) ** 2 +
                                           (t.row - unit.row) ** 2))
        return best

    def _resolve_combat(self) -> None:
        """Resolve all unit-to-unit and unit-to-tower combat."""
        all_units = self.player_units + self.opponent_units
        alive = [u for u in all_units if u.is_alive]

        # Group by proximity
        for unit in alive:
            if unit.is_building:
                continue  # Buildings attack in _tower_attacks

            enemy_units = (self.opponent_units if unit.owner == "player"
                           else self.player_units)
            enemies_in_range = [e for e in enemy_units if e.is_alive
                                and self._distance(unit, e) <= unit.range]

            if not enemies_in_range:
                continue

            # Pick nearest enemy in range
            target = min(enemies_in_range,
                         key=lambda e: self._distance(unit, e))

            # Deal damage
            target.take_damage(unit.damage)

    def _tower_attacks(self) -> None:
        """Process tower attacks."""
        all_towers = self.player_towers + self.opponent_towers
        for tower in all_towers:
            if not tower.is_alive:
                continue

            enemy_units = (self.opponent_units if tower.owner == "player"
                           else self.player_units)
            enemies_in_range = [e for e in enemy_units if e.is_alive
                                and self._distance(tower, e) <= tower.range]

            if not enemies_in_range:
                continue

            # Attack nearest
            target = min(enemies_in_range,
                         key=lambda e: self._distance(tower, e))
            target.take_damage(tower.damage)

    def _distance(self, a: UnitState, b: UnitState) -> float:
        """Euclidean distance between two units."""
        return math.sqrt((a.col - b.col) ** 2 + (a.row - b.row) ** 2)

    def _cycle_hand(self, player: str) -> None:
        """Cycle the card hand for a player (every 3 ticks)."""
        if player == "player":
            hand = self.player_hand
            queue = self.player_deck_queue
        else:
            hand = self.opponent_hand
            queue = self.opponent_deck_queue

        # Find the first non-cooldown slot and replace it
        for i in range(4):
            if len(queue) == 0:
                break
            if hand[i] is not None and len(queue) > 0:
                # Swap: put current card back, draw new one
                queue.append(hand[i])
                hand[i] = queue.pop(0)

    def _check_win_conditions(self) -> SimulationStepResult:
        """Check if the game has ended and compute rewards."""
        # King tower destruction = instant win
        player_king = next((t for t in self.player_towers
                            if t.unit_type == "tower" and not t.is_building), None)
        opp_king = next((t for t in self.opponent_towers
                         if t.unit_type == "tower" and not t.is_building), None)

        if player_king and not player_king.is_alive:
            self.terminated = True
            self.opponent_trophies += 2
            return SimulationStepResult(
                rewards={"player": -1.0, "opponent": 1.0},
                terminated=True,
                info={"winner": "opponent", "reason": "king_tower_destroyed"},
            )

        if opp_king and not opp_king.is_alive:
            self.terminated = True
            self.player_trophies += 2
            return SimulationStepResult(
                rewards={"player": 1.0, "opponent": -1.0},
                terminated=True,
                info={"winner": "player", "reason": "king_tower_destroyed"},
            )

        # Check overtime
        if not self.is_overtime and self.tick >= self.match_duration_ticks:
            self.is_overtime = True
            self.truncated = False
            return SimulationStepResult(
                rewards={"player": 0.0, "opponent": 0.0},
                terminated=False,
                truncated=True,
                info={"phase": "overtime"},
            )

        # Time's up
        total_ticks = self.match_duration_ticks + self.overtime_ticks
        if self.tick >= total_ticks:
            self.terminated = True
            # Determine winner by trophy count
            if self.player_trophies > self.opponent_trophies:
                self.player_trophies += 2
                return SimulationStepResult(
                    rewards={"player": 1.0, "opponent": -1.0},
                    terminated=True,
                    info={"winner": "player", "reason": "time_up"},
                )
            elif self.opponent_trophies > self.player_trophies:
                self.opponent_trophies += 2
                return SimulationStepResult(
                    rewards={"player": -1.0, "opponent": 1.0},
                    terminated=True,
                    info={"winner": "opponent", "reason": "time_up"},
                )
            else:
                # Tie
                return SimulationStepResult(
                    rewards={"player": 0.0, "opponent": 0.0},
                    terminated=True,
                    info={"winner": "tie", "reason": "time_up"},
                )

        # Compute shaped rewards during gameplay
        player_towers_alive = sum(1 for t in self.player_towers if t.is_alive)
        opp_towers_alive = sum(1 for t in self.opponent_towers if t.is_alive)

        # Shaped reward: tower damage and progress
        player_reward = (
            0.3 * (opp_towers_alive - player_towers_alive) / 6.0  # Tower progress
            + 0.2 * self.player_elixir / self.elixir_max  # Elixir efficiency
            + 0.1 * (sum(u.hp for u in self.player_units if u.is_alive) /
                     max(sum(u.max_hp for u in self.player_units), 1))  # Unit survival
        )

        opponent_reward = (
            0.3 * (player_towers_alive - opp_towers_alive) / 6.0
            + 0.2 * self.opponent_elixir / self.elixir_max
            + 0.1 * (sum(u.hp for u in self.opponent_units if u.is_alive) /
                     max(sum(u.max_hp for u in self.opponent_units), 1))
        )

        return SimulationStepResult(
            rewards={"player": player_reward, "opponent": opponent_reward},
            terminated=False,
            truncated=False,
            info={
                "player_towers_alive": player_towers_alive,
                "opponent_towers_alive": opp_towers_alive,
                "player_elixir": self.player_elixir,
                "opponent_elixir": self.opponent_elixir,
            },
        )
