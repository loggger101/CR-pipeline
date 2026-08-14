"""Core simulation engine for Clash Royale.

Tick-based game loop that processes:
1. Elixir regeneration
2. Unit spawning and card cycling
3. Status effect updates (stun, slow, poison)
4. Unit pathfinding toward targets (bridge crossing, lane-based)
5. Unit-to-unit and unit-to-tower combat
6. Tower attacks
7. Spell effects (damage, stun, slow, poison, heal, tornado)
8. King tower activation
9. Unit death effects (elixir golem split, wall breakers charge)
10. Win/loss detection
11. Trophy/crown reward calculation

Designed for speed: each tick processes game events in pure Python/NumPy, and
dead units are reclaimed every tick so per-tick work stays proportional to the
number of live units rather than to everything ever spawned.

Timebase
--------
``TICKS_PER_SECOND = 10``, so the default 1800-tick regulation is 180 seconds
of game time. Card stats are authored in real units -- ``attack_speed`` in
seconds, ``move_speed`` in tiles per second -- and converted through that
constant rather than being consumed per-tick directly.

Key mechanics
-------------
- Princess towers stand closer to the river than their king; the king starts
  inactive and wakes on damage or on losing a princess tower.
- Crowns accrue as towers fall (1 per princess, 3 for the king). A king kill
  ends the match; otherwise crowns decide it at time, via overtime if level.
- The deck is a fixed 8-card rotation with 4 in hand; playing a card cycles
  only that slot.
- Troops deploy on their owner's half only; spells target the whole arena.
"""

from __future__ import annotations

import copy
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from .entities import (
    CARD_DEFS, TOWER_DEFS, DeploymentZone, EntityType,
    TargetingMode, TowerDefinition, UnitInstance, UnitType,
)
from .actions import Action, ActionType
from .state import (
    GameStateSnapshot, UnitState, UnitStatus, compute_state_from_arena,
)

logger = logging.getLogger(__name__)

# UnitStatus lives in state.py so UnitState can branch on the same members the
# engine applies. Re-exported here for backwards compatibility.
__all__ = [
    "SimulationEngine", "SimulationStepResult", "OpponentStrategy",
    "ReplayFrame", "UnitStatus",
]


@dataclass
class SimulationStepResult:
    """Result of a single simulation step.

    Attributes:
        rewards: Dict of reward signals for each player.
        terminated: Whether the game has ended.
        truncated: Whether the game was truncated (e.g., timeout).
        info: Additional diagnostic information.
        crown_rewards: Dict of crown rewards per player.
        tower_damage_dealt: Dict of total tower damage dealt per player.
    """
    rewards: Dict[str, float] = field(default_factory=lambda: {"player": 0.0, "opponent": 0.0})
    terminated: bool = False
    truncated: bool = False
    info: Dict = field(default_factory=dict)
    crown_rewards: Dict[str, int] = field(default_factory=lambda: {"player": 0, "opponent": 0})
    tower_damage_dealt: Dict[str, float] = field(default_factory=lambda: {"player": 0.0, "opponent": 0.0})


class OpponentStrategy(Enum):
    """Opponent AI strategies for simulation."""
    RANDOM = auto()
    GREEDY = auto()
    BALANCED = auto()
    AGGRESSIVE = auto()
    DEFENSIVE = auto()
    SELF_PLAY = auto()
    DECK_AWARE = auto()


@dataclass
class ReplayFrame:
    """A single frame in a replay for post-match analysis."""
    tick: int
    player_units: List[dict] = field(default_factory=list)
    opponent_units: List[dict] = field(default_factory=list)
    player_towers: List[dict] = field(default_factory=list)
    opponent_towers: List[dict] = field(default_factory=list)
    player_elixir: float = 5.0
    opponent_elixir: float = 5.0
    player_trophies: int = 0
    opponent_trophies: int = 0
    player_hand: List[str] = field(default_factory=list)
    opponent_hand: List[str] = field(default_factory=list)
    player_towers_destroyed: int = 0
    opponent_towers_destroyed: int = 0
    is_overtime: bool = False
    arena_state: Optional[np.ndarray] = None


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

    Enhanced with:
    - Multiple opponent AI strategies
    - Proper card cycling
    - Expanded card registry
    - Unit death effects
    - Enhanced rewards
    - Replay data
    """

    # Arena dimensions
    GRID_COLS = 8
    GRID_ROWS = 6

    # Bridge columns
    BRIDGE_COLS = [3, 4]

    # Bridge row (where bridges are located)
    BRIDGE_ROW = 2

    # King tower activation range
    KING_ACTIVATION_RANGE = 4.0

    # Simulation timebase. A match of 1800 ticks is 180s of game time, matching
    # a real Clash Royale regulation match. Card stats are authored in seconds
    # (attack_speed) and tiles-per-second (move_speed), so both are converted
    # through this constant rather than being consumed per-tick directly.
    TICKS_PER_SECOND = 10

    # Real Clash Royale regenerates 1 elixir every 2.8s in single elixir.
    DEFAULT_ELIXIR_REGEN_PER_TICK = 1.0 / 2.8 / TICKS_PER_SECOND

    # Poison/earthquake damage is authored per second and scaled by tick.
    POISON_DPS = 90.0

    # Tower rows. Princess towers sit closer to the river than the king so that
    # attackers meet them first; the king is the deepest structure on each side.
    OPPONENT_KING_ROW = 0.0
    OPPONENT_PRINCESS_ROW = 1.0
    PLAYER_KING_ROW = 5.0
    PLAYER_PRINCESS_ROW = 4.0
    PRINCESS_COLS = (2.0, 5.0)
    KING_COL = 3.5

    # Default deck for training
    DEFAULT_DECK = [
        "knight", "archers", "fireball", "musketeer",
        "mini_pekka", "valkyrie", "baby_dragon", "wizard",
    ]

    # Common training decks for varied opponent behavior
    TRAINING_DECKS = {
        "hog_cycle": ["hog_rider", "skeletons", "ice_golem", "fireball",
                      "goblin", "knight", "archers", "zap"],
        "golem_push": ["golem", "baby_dragon", "mega_minion", "lightning",
                       "skeleton_barrel", "mini_pekka", "electro_wizard", "tornado"],
        "double_prince": ["prince", "royal_giant", "wizard", "fireball",
                          "baby_dragon", "valkyrie", "musketeer", "zap"],
        "minion_horde": ["minion_horde", "balloon", "wizard", "fireball",
                         "ice_golem", "valkyrie", "musketeer", "zap"],
        "bait": ["goblin_barrel", "princess", "knight", "archers",
                 "fireball", "mini_pekka", "valkyrie", "zap"],
        "lalo": ["lumberjack", "balloon", "skeleton_barrel", "tornado",
                 "fireball", "mega_minion", "electro_wizard", "valkyrie"],
        "xbow": ["x_bow", "skeletons", "ice_golem", "fireball",
                 "knight", "archers", "valkyrie", "zap"],
        "giant_bone": ["giant", "baby_dragon", "mega_minion", "lightning",
                       "skeleton_barrel", "mini_pekka", "electro_wizard", "tornado"],
    }

    # Deploy cooldown in ticks
    DEPLOY_COOLDOWN = 3

    def __init__(
        self,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
        match_duration_ticks: int = 1800,
        overtime_ticks: int = 120,
        elixir_regen_rate: Optional[float] = None,
        elixir_max: int = 10,
        double_elixir_overtime: bool = True,
        seed: Optional[int] = None,
        opponent_strategy: OpponentStrategy = OpponentStrategy.BALANCED,
        record_replay: bool = True,
    ):
        """Initialize the simulation engine.

        Args:
            deck: Player's card deck (8 cards). Defaults to DEFAULT_DECK.
            opponent_deck: Opponent's card deck. Defaults to a shuffled DEFAULT_DECK.
            match_duration_ticks: Length of regulation match.
            overtime_ticks: Additional ticks for overtime.
            elixir_regen_rate: Elixir regenerated per tick. Defaults to the
                real-game rate (1 elixir per 2.8 seconds).
            elixir_max: Maximum elixir capacity.
            double_elixir_overtime: Whether elixir doubles in overtime.
            seed: Random seed for reproducibility.
            opponent_strategy: AI strategy for the opponent.
            record_replay: Whether to record replay data.
        """
        self.rng = np.random.RandomState(seed)
        self.match_duration_ticks = match_duration_ticks
        self.overtime_ticks = overtime_ticks
        self.elixir_regen_rate = (
            self.DEFAULT_ELIXIR_REGEN_PER_TICK if elixir_regen_rate is None
            else elixir_regen_rate
        )
        self.elixir_max = elixir_max
        self.double_elixir_overtime = double_elixir_overtime
        self.opponent_strategy = opponent_strategy
        self.record_replay = record_replay

        # Set up decks
        if deck is not None:
            self.player_deck = list(deck)
        else:
            self.player_deck = list(self.DEFAULT_DECK)
        if opponent_deck is not None:
            self.opponent_deck = list(opponent_deck)
        else:
            # Pick a random training deck
            deck_name = self.rng.choice(list(self.TRAINING_DECKS.keys()))
            self.opponent_deck = list(self.TRAINING_DECKS[deck_name])
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

        # Tower damage tracking
        self._tower_damage_dealt: Dict[str, float] = {"player": 0.0, "opponent": 0.0}

        # Cumulative units spawned per side (corpses are reclaimed each tick)
        self._units_spawned: Dict[str, int] = {"player": 0, "opponent": 0}

        # Arena grid: 0 = empty, > 0 = occupied (for placement validation)
        self.arena: np.ndarray = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=np.int32)

        # Action history for replay
        self.action_history: List[dict] = []

        # Replay data (full state snapshots)
        self.replay_frames: List[ReplayFrame] = []

        # Initial state for reset
        self._initial_deck = list(self.player_deck)
        self._initial_opponent_deck = list(self.opponent_deck)
        self._initial_seed = seed
        self._initial_opponent_strategy = opponent_strategy

        self.reset()

    def reset(self, seed: Optional[int] = None,
              opponent_deck: Optional[List[str]] = None) -> GameStateSnapshot:
        """Reset to initial state and return the first observation.

        Args:
            seed: Reseed the engine for this episode. Without it every reset
                replays the identical match, which made evaluating an agent
                over N matches no more informative than evaluating it over one.
            opponent_deck: Swap the opponent's deck for this episode. Lets a
                single engine evaluate an agent across several archetypes
                without paying to rebuild it.

        Returns:
            Initial GameStateSnapshot.
        """
        if seed is not None:
            self._initial_seed = seed
        if opponent_deck is not None:
            self._initial_opponent_deck = list(opponent_deck)
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
        self.replay_frames = []
        self._tower_damage_dealt = {"player": 0.0, "opponent": 0.0}
        self._units_spawned = {"player": 0, "opponent": 0}

        self.player_trophies = 0
        self.opponent_trophies = 0
        self.player_towers_destroyed = 0
        self.opponent_towers_destroyed = 0

        # Place towers
        self._place_towers()

        # Record initial replay frame
        if self.record_replay:
            self._record_replay_frame()

        return self._get_state()

    def _place_towers(self) -> None:
        """Place all towers on the arena.

        Every tower is a building: stationary, attacking through
        ``_tower_attacks`` only. The king additionally starts inactive and is
        placed behind its princess towers, so attackers meet a princess first.
        """
        layouts = [
            ("opponent", self.opponent_towers, self.opponent_units, [
                ("p2_princess_left", self.PRINCESS_COLS[0], self.OPPONENT_PRINCESS_ROW),
                ("p2_princess_right", self.PRINCESS_COLS[1], self.OPPONENT_PRINCESS_ROW),
                ("p2_king", self.KING_COL, self.OPPONENT_KING_ROW),
            ]),
            ("player", self.player_towers, self.player_units, [
                ("p1_princess_left", self.PRINCESS_COLS[0], self.PLAYER_PRINCESS_ROW),
                ("p1_princess_right", self.PRINCESS_COLS[1], self.PLAYER_PRINCESS_ROW),
                ("p1_king", self.KING_COL, self.PLAYER_KING_ROW),
            ]),
        ]

        for owner, tower_list, unit_list, positions in layouts:
            for name, col, row in positions:
                defn = TOWER_DEFS[name]
                unit = UnitState(
                    unit_type="tower",
                    owner=owner,
                    hp=defn.hitpoints,
                    max_hp=defn.hitpoints,
                    col=col,
                    row=row,
                    is_alive=True,
                    is_building=True,
                    speed=0.0,
                    base_speed=0.0,
                    range=defn.attack_range,
                    damage=defn.damage,
                    attack_speed=defn.attack_speed,
                    can_target_air=True,
                    can_target_ground=True,
                    is_king=defn.is_king_tower,
                    # Princess towers are always live; the king must be
                    # activated by damage or by losing a princess tower.
                    is_active=not defn.is_king_tower,
                    activation_range=getattr(
                        defn, "activation_range", self.KING_ACTIVATION_RANGE),
                )
                tower_list.append(unit)
                unit_list.append(unit)
                self.arena[int(row), int(col)] = 1  # Occupied

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

        # 3. Update status effects (stun, slow, poison)
        self._update_status_effects()

        # 4. Elixir regeneration
        self._regenerate_elixir()

        # 5. Card cooldown countdown
        self._countdown_cooldowns()

        # 6. Unit pathfinding and movement
        self._move_units()

        # 7. Combat resolution (unit vs unit)
        self._resolve_combat()

        # 8. Tower attacks
        self._tower_attacks()

        # 9. Process unit death effects (elixir golem split, wall breakers)
        self._process_death_effects()

        # 10. Drop corpses so per-tick loops stay proportional to live units
        self._reclaim_dead_units()

        # 11. Check win conditions
        result = self._check_win_conditions()

        # 12. Tick counter
        self.tick += 1

        # 13. Record replay frame (every 10 ticks)
        if self.record_replay and self.tick % 10 == 0:
            self._record_replay_frame()

        return result

    def _process_action(self, action: Action, player: str) -> None:
        """Process a player's action (card deployment)."""
        if action.action_type not in (ActionType.DEPLOY_UNIT, ActionType.DEPLOY_SPELL):
            return

        hand = self.player_hand if player == "player" else self.opponent_hand
        cooldowns = self.player_cooldowns if player == "player" else self.opponent_cooldowns
        elixir = self.player_elixir if player == "player" else self.opponent_elixir

        card_idx = action.card_index
        if card_idx is None or card_idx < 0 or card_idx >= len(hand):
            return

        card_name = hand[card_idx]
        card_def = CARD_DEFS.get(card_name)
        if card_def is None:
            return

        # Check cooldown
        if cooldowns[card_idx] > 0:
            return

        # Check if card is silenced (for spell cards)
        if card_def.card_type == "spell" and self._is_silenced(player):
            return

        # Check elixir
        if elixir < card_def.cost:
            return

        # Check deployment zone
        target_col = action.target_col
        target_row = action.target_row

        if target_col is None or target_row is None:
            return

        if not self._is_valid_placement(card_def, target_col, target_row, player):
            return

        # Deploy the card
        self._deploy_card(card_name, target_col, target_row, player, card_idx)

        # Deduct elixir, start cooldown, and cycle this slot to the next card
        cooldown_value = self.DEPLOY_COOLDOWN
        if player == "player":
            self.player_elixir = max(0, self.player_elixir - card_def.cost)
            self.player_cooldowns[card_idx] = cooldown_value
        else:
            self.opponent_elixir = max(0, self.opponent_elixir - card_def.cost)
            self.opponent_cooldowns[card_idx] = cooldown_value

        self._cycle_played_card(player, card_idx)

        # Record action
        self.action_history.append({
            "tick": self.tick,
            "player": player,
            "card": card_name,
            "col": target_col,
            "row": target_row,
        })

    def _is_valid_placement(self, card_def, col: float, row: float,
                            player: str) -> bool:
        """Check whether a card may be placed at the given arena position.

        Units may only be dropped on their owner's half. Spells target the
        whole arena -- restricting them to the caster's own half (as the
        engine previously did for every card type) made every damage spell
        unable to reach the towers and units it exists to hit.
        """
        if col < 0 or col > self.GRID_COLS - 1:
            return False
        if row < 0 or row > self.GRID_ROWS - 1:
            return False

        if card_def.card_type == "spell":
            return True

        if player == "player":
            return row >= self.BRIDGE_ROW + 1
        return row <= self.BRIDGE_ROW

    def _is_silenced(self, player: str) -> bool:
        """Check if a player is silenced (cannot deploy spells)."""
        units = self.player_units if player == "player" else self.opponent_units
        for unit in units:
            if unit.is_alive and unit.status == UnitStatus.SILENCED:
                return True
        return False

    def _deploy_card(self, card_name: str, col: float, row: float,
                     player: str, hand_idx: int) -> None:
        """Deploy a card at the given position."""
        card_def = CARD_DEFS[card_name]

        if card_def.card_type == "spell":
            # Handle spell by type
            spell_type = card_def.spell_type
            
            if spell_type == "damage":
                self._apply_spell_damage(col, row, card_def.spell_damage, card_def.spell_radius,
                                         player, card_name)
                # Secondary effects for damage spells
                if card_name == "zap":
                    self._apply_stun(col, row, card_def.spell_radius, player)
            elif spell_type == "poison":
                self._apply_poison(col, row, card_def.spell_radius, player)
            elif spell_type == "stun":
                self._apply_stun(col, row, card_def.spell_radius, player)
            elif spell_type == "heal":
                self._apply_heal(col, row, card_def.spell_radius, player)
            elif spell_type == "tornado":
                self._apply_tornado(col, row, card_def.spell_radius, player)
            elif spell_type == "earthquake":
                self._apply_earthquake(col, row, card_def.spell_damage, card_def.spell_radius,
                                       card_def.spell_duration, player)
            elif spell_type == "graveyard":
                self._apply_graveyard(col, row, card_def.spawn_count, card_def.spawned_unit, player)
            elif spell_type == "deliver":
                self._apply_royal_deliver(col, row, card_def.spawned_unit, player)
            else:
                # Default: apply raw damage
                if card_def.spell_damage > 0:
                    self._apply_spell_damage(col, row, card_def.spell_damage, card_def.spell_radius,
                                             player, card_name)
            return

        # A swarm card deploys as several copies of another unit (Minions,
        # Skeleton Army). Cards that name a spawned_unit but deploy as one body
        # (Lava Hound, Night Witch) must deploy *themselves*; treating any
        # spawned_unit as a swarm replaced those tanks with their spawn.
        if card_def.spawn_count > 1 and card_def.spawned_unit:
            for _ in range(card_def.spawn_count):
                self._spawn_unit(
                    card_def.spawned_unit,
                    col + self.rng.uniform(-0.3, 0.3),
                    row + self.rng.uniform(-0.3, 0.3),
                    player,
                )
        else:
            self._spawn_unit(card_name, col, row, player)

    def _spawn_unit(self, card_name: str, col: float, row: float,
                    player: str) -> None:
        """Spawn a unit on the arena.

        ``move_speed`` is authored in tiles per second, so it is converted to
        the per-tick step the movement code actually applies.
        """
        card_def = CARD_DEFS.get(card_name)
        if card_def is None:
            logger.warning("Unknown card %r requested for spawn; ignoring", card_name)
            return

        is_building = card_def.unit_type == UnitType.BUILDING
        speed_per_tick = (
            0.0 if is_building
            else card_def.move_speed / self.TICKS_PER_SECOND
        )
        is_air = card_def.unit_type == UnitType.AIR

        unit = UnitState(
            unit_type=card_name,
            owner=player,
            hp=card_def.hitpoints,
            max_hp=card_def.hitpoints,
            col=max(0.0, min(self.GRID_COLS - 1, col)),
            row=max(0.0, min(self.GRID_ROWS - 1, row)),
            is_alive=True,
            is_building=is_building,
            speed=speed_per_tick,
            base_speed=speed_per_tick,
            range=card_def.attack_range,
            damage=card_def.damage,
            attack_speed=card_def.attack_speed,
            target_pref=(card_def.targeting_mode or TargetingMode.NEAREST),
            is_air=is_air,
            is_ground=not is_air,
            # What this unit may attack is taken from the card definition for
            # air and ground units alike (minions, for instance, are air units
            # that still hit ground targets).
            can_target_air=bool(card_def.target_air),
            can_target_ground=bool(card_def.target_ground),
            is_spawned=True,
            spawn_tick=self.tick,
        )

        if player == "player":
            self.player_units.append(unit)
        else:
            self.opponent_units.append(unit)
        self._units_spawned[player] += 1

    def _enemies_of(self, caster: str) -> List[UnitState]:
        """Live units belonging to the side opposing ``caster``."""
        units = self.opponent_units if caster == "player" else self.player_units
        return [u for u in units if u.is_alive]

    def _allies_of(self, caster: str) -> List[UnitState]:
        """Live units belonging to ``caster``."""
        units = self.player_units if caster == "player" else self.opponent_units
        return [u for u in units if u.is_alive]

    def _units_in_radius(self, units: List[UnitState], center_col: float,
                         center_row: float, radius: float,
                         hits_air: bool = True,
                         hits_ground: bool = True) -> List[UnitState]:
        """Select units within ``radius`` that the effect is allowed to hit.

        Air/ground eligibility is a property of the *effect*, not of the unit
        being hit. The previous implementation tested the victim's own
        ``can_target_air``/``can_target_ground`` (which describe what that unit
        may attack), so spell coverage was effectively arbitrary.
        """
        hits = []
        for unit in units:
            if unit.is_air and not hits_air:
                continue
            if unit.is_ground and not hits_ground:
                continue
            dist = math.hypot(unit.col - center_col, unit.row - center_row)
            if dist <= radius:
                hits.append(unit)
        return hits

    def _damage_unit(self, unit: UnitState, damage: float,
                     attacker_side: str) -> float:
        """Apply damage, book tower damage, and run death handling once."""
        actual = unit.take_damage(damage)
        if unit.is_building and actual > 0:
            # Booked against the tower's owner: "damage taken by this side".
            self._tower_damage_dealt[unit.owner] += actual
            # A king tower also wakes when it is damaged directly.
            if unit.is_king and unit.is_alive:
                unit.is_active = True
        if unit.just_died:
            self._handle_unit_death(unit)
        return actual

    def _apply_spell_damage(self, center_col: float, center_row: float,
                            damage: float, radius: float, caster: str,
                            spell_name: str = "") -> None:
        """Apply spell damage to all enemy units within radius of center."""
        hits_air = True
        hits_ground = True
        if spell_name == "earthquake":
            hits_air = False  # Earthquake only shakes the ground
        for unit in self._units_in_radius(
                self._enemies_of(caster), center_col, center_row, radius,
                hits_air=hits_air, hits_ground=hits_ground):
            self._damage_unit(unit, damage, caster)

    def _apply_stun(self, center_col: float, center_row: float,
                    radius: float, caster: str,
                    duration_s: float = 0.5) -> None:
        """Apply stun effect to all enemy units within radius."""
        ticks = max(1, int(duration_s * self.TICKS_PER_SECOND))
        for unit in self._units_in_radius(
                self._enemies_of(caster), center_col, center_row, radius):
            if unit.is_building:
                continue  # Towers cannot be stunned
            unit.apply_status(UnitStatus.STUNNED, ticks, "stun")

    def _apply_poison(self, center_col: float, center_row: float,
                      radius: float, caster: str,
                      duration_s: float = 8.0) -> None:
        """Apply poison (damage per tick for a duration) to enemies in radius."""
        ticks = max(1, int(duration_s * self.TICKS_PER_SECOND))
        for unit in self._units_in_radius(
                self._enemies_of(caster), center_col, center_row, radius):
            unit.apply_status(UnitStatus.POISONED, ticks, "poison")

    def _apply_heal(self, center_col: float, center_row: float,
                    radius: float, caster: str,
                    heal_amount: float = 300.0) -> None:
        """Heal friendly units within radius."""
        for unit in self._units_in_radius(
                self._allies_of(caster), center_col, center_row, radius):
            if unit.is_building:
                continue
            unit.heal(heal_amount)

    def _apply_tornado(self, center_col: float, center_row: float,
                       radius: float, caster: str) -> None:
        """Pull enemy units toward the tornado centre and slow them."""
        ticks = max(1, int(3.0 * self.TICKS_PER_SECOND))
        for unit in self._units_in_radius(
                self._enemies_of(caster), center_col, center_row, radius):
            if unit.is_building:
                continue  # Buildings cannot be moved
            dist = math.hypot(center_col - unit.col, center_row - unit.row)
            if dist > 0.1:
                pull_strength = 0.3
                unit.col += ((center_col - unit.col) / dist) * pull_strength
                unit.row += ((center_row - unit.row) / dist) * pull_strength
            unit.apply_status(UnitStatus.SLOWED, ticks, "tornado")

    def _apply_earthquake(self, center_col: float, center_row: float,
                          damage: float, radius: float, duration: float,
                          caster: str) -> None:
        """Earthquake: ground-only burst plus a lingering damage-over-time."""
        ticks = max(1, int(duration * self.TICKS_PER_SECOND))
        for unit in self._units_in_radius(
                self._enemies_of(caster), center_col, center_row, radius,
                hits_air=False):
            self._damage_unit(unit, damage, caster)
            if unit.is_alive and not unit.is_building:
                unit.apply_status(UnitStatus.POISONED, ticks, "earthquake")

    def _apply_graveyard(self, center_col: float, center_row: float,
                         spawn_count: int, spawned_unit: str, caster: str) -> None:
        """Apply graveyard spell - spawns skeletons at target location.
        
        Graveyard spawns skeleton units at the target location on opponent's side.
        """
        if not spawned_unit:
            return
        # Spawn skeletons at the target location (slightly spread out)
        for i in range(spawn_count):
            angle = (2 * math.pi * i) / spawn_count
            offset_x = math.cos(angle) * 0.3
            offset_y = math.sin(angle) * 0.3
            spawn_col = center_col + offset_x
            spawn_row = center_row + offset_y
            self._spawn_unit(spawned_unit, spawn_col, spawn_row, caster)

    def _apply_royal_deliver(self, center_col: float, center_row: float,
                             spawned_unit: str, caster: str) -> None:
        """Apply Royal Deliver - spawns a royal ghost at target location.
        
        Royal Deliver is a spell that summons a Royal Ghost at the target.
        The ghost becomes invisible after attacking once.
        """
        if not spawned_unit:
            return
        self._spawn_unit(spawned_unit, center_col, center_row, caster)

    def _process_death_effects(self) -> None:
        """Process unit death effects (elixir golem split, wall breakers charge).

        Runs once per dead unit: the ``just_died`` flag is cleared as each
        corpse is handled. Iterating a single combined list also avoids the
        double-processing the previous nested loop performed.
        """
        for unit in list(self.player_units) + list(self.opponent_units):
            if not unit.just_died:
                continue
            unit.just_died = False

            card_def = CARD_DEFS.get(unit.unit_type)
            if card_def is None:
                continue

            # Death spawns, declared on the card (Golem, Lava Hound). The
            # previous version hardcoded "elixir_golem"/"mini_golem", neither
            # of which exists in the registry, so no card ever split.
            if card_def.death_spawn_count > 0 and card_def.death_spawned_unit:
                for _ in range(card_def.death_spawn_count):
                    self._spawn_unit(
                        card_def.death_spawned_unit,
                        unit.col + self.rng.uniform(-0.5, 0.5),
                        unit.row + self.rng.uniform(-0.5, 0.5),
                        unit.owner,
                    )

            # Wall breakers detonate on the nearest structure.
            if unit.unit_type in ("wall_breakers", "wall_breaker_mini"):
                target = self._find_nearest_building(unit.col, unit.row, unit.owner)
                if target is not None:
                    self._damage_unit(target, 400, unit.owner)

    def _reclaim_dead_units(self) -> None:
        """Drop dead non-tower units from the per-tick iteration lists.

        Corpses were previously retained for the whole match, so every
        movement/combat pass scanned them and the unit-survival reward divided
        by the max HP of units that had long since died. Towers are kept
        (dead or alive) because win conditions and rendering read them.
        """
        self.player_units = [u for u in self.player_units
                             if u.is_alive or u.is_building]
        self.opponent_units = [u for u in self.opponent_units
                               if u.is_alive or u.is_building]

    def _find_nearest_building(self, col: float, row: float,
                                exclude_owner: str) -> Optional[UnitState]:
        """Find the nearest building for wall breakers to target."""
        targets = (self.player_towers if exclude_owner == "player"
                   else self.opponent_towers)
        alive_buildings = [t for t in targets if t.is_alive]
        if not alive_buildings:
            return None
        return min(alive_buildings, key=lambda t: math.sqrt((t.col - col) ** 2 + (t.row - row) ** 2))

    def _update_status_effects(self) -> None:
        """Tick down status effects and apply damage-over-time."""
        poison_per_tick = self.POISON_DPS / self.TICKS_PER_SECOND
        for unit in list(self.player_units) + list(self.opponent_units):
            if not unit.is_alive:
                continue

            if unit.update_status():
                continue  # Expired this tick

            if unit.status == UnitStatus.POISONED:
                attacker = "opponent" if unit.owner == "player" else "player"
                self._damage_unit(unit, poison_per_tick, attacker)

    def _countdown_cooldowns(self) -> None:
        """Decrement card cooldowns."""
        self.player_cooldowns = np.maximum(self.player_cooldowns - 1, 0)
        self.opponent_cooldowns = np.maximum(self.opponent_cooldowns - 1, 0)

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
        """Move all units toward their targets using lane-based pathfinding."""
        for unit in list(self.player_units) + list(self.opponent_units):
            if not unit.is_alive or unit.is_building:
                continue

            # Stunned units cannot move (speed is also zeroed by the status).
            if unit.status == UnitStatus.STUNNED or unit.speed <= 0:
                continue

            target = self._find_target(unit)
            if target is None:
                continue

            # Already close enough to attack: hold position and let combat
            # resolve. Previously units closed to within 0.1 tiles of every
            # target, so ranged troops walked into melee before firing.
            dist = self._distance(unit, target)
            if dist <= unit.range:
                continue

            # Ground units must use a bridge to cross the river; air units fly.
            if not unit.is_air and not self._can_cross_bridge(unit, target):
                bridge_col = self._find_nearest_bridge(unit.col)
                if bridge_col is not None:
                    self._move_toward(unit, float(bridge_col),
                                      float(self.BRIDGE_ROW), unit.speed)
                continue

            self._move_toward(unit, target.col, target.row, unit.speed)

            new_dist = self._distance(unit, target)
            if new_dist < dist:
                unit.path_progress = max(0.0, min(1.0, unit.path_progress + 0.01))

    def _move_toward(self, unit: UnitState, target_col: float,
                     target_row: float, speed: float) -> None:
        """Move a unit toward a specific point."""
        dx = target_col - unit.col
        dy = target_row - unit.row
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist <= 0.1:
            return

        move_x = (dx / dist) * min(speed, dist)
        move_y = (dy / dist) * min(speed, dist)

        unit.col += move_x
        unit.row += move_y

        unit.col = max(0, min(self.GRID_COLS - 1, unit.col))
        unit.row = max(0, min(self.GRID_ROWS - 1, unit.row))

    def _can_cross_bridge(self, unit: UnitState, target: UnitState) -> bool:
        """Check if a unit can cross the bridge to reach its target."""
        # If already on the correct side, no bridge needed
        if unit.owner == "player" and target.row <= unit.row:
            # Player unit needs to go toward opponent (up)
            if unit.row > self.BRIDGE_ROW:
                # Must cross a bridge
                return self._bridge_open_at(unit.col)
        elif unit.owner == "opponent" and target.row >= unit.row:
            # Opponent unit needs to go toward player (down)
            if unit.row < self.BRIDGE_ROW:
                # Must cross a bridge
                return self._bridge_open_at(unit.col)
        return True

    # How far either side of a bridge column counts as "on the bridge".
    BRIDGE_TOLERANCE = 0.75

    def _bridge_open_at(self, col: float) -> bool:
        """Check whether ``col`` lines up with a bridge crossing."""
        return any(abs(col - bridge_col) <= self.BRIDGE_TOLERANCE
                   for bridge_col in self.BRIDGE_COLS)

    def _find_nearest_bridge(self, col: float) -> Optional[float]:
        """Find the nearest open bridge column."""
        nearest = None
        min_dist = float('inf')
        for bridge_col in self.BRIDGE_COLS:
            dist = abs(col - bridge_col)
            if dist < min_dist:
                min_dist = dist
                nearest = bridge_col
        return nearest

    def _can_attack(self, attacker: UnitState, victim: UnitState) -> bool:
        """Whether ``attacker`` is permitted to damage ``victim``."""
        if victim.is_air and not attacker.can_target_air:
            return False
        if victim.is_ground and not attacker.can_target_ground:
            return False
        return True

    def _find_target(self, unit: UnitState) -> Optional[UnitState]:
        """Find the best target for a unit to approach and attack.

        Building-targeting troops (``NEAREST_TOWER``) ignore defenders and head
        for structures; everything else takes the nearest thing it can legally
        damage. Air/ground eligibility is enforced here so units never path
        toward a target they could not hit on arrival.
        """
        if unit.owner == "player":
            enemy_towers, enemy_units = self.opponent_towers, self.opponent_units
        else:
            enemy_towers, enemy_units = self.player_towers, self.player_units

        towers = [t for t in enemy_towers
                  if t.is_alive and self._can_attack(unit, t)]
        troops = [u for u in enemy_units
                  if u.is_alive and not u.is_building
                  and self._can_attack(unit, u)]

        if unit.target_pref == TargetingMode.NEAREST_TOWER:
            candidates = towers or troops
        else:
            candidates = towers + troops

        if not candidates:
            return None
        return min(candidates, key=lambda t: self._distance(unit, t))

    def _attack_cooldown_ticks(self, attacker: UnitState) -> int:
        """Ticks that must elapse between attacks.

        ``attack_speed`` is authored in seconds, so it is scaled by the
        timebase. The previous ``60 * attack_speed`` treated it as a 60Hz
        clock, leaving ~72 ticks (7+ seconds) between swings.
        """
        return max(1, int(round(attacker.attack_speed * self.TICKS_PER_SECOND)))

    def _perform_attack(self, attacker: UnitState,
                        candidates: List[UnitState]) -> None:
        """Attack the nearest legal target in range, if the attacker is ready."""
        if self.tick - attacker.last_attack_tick < self._attack_cooldown_ticks(attacker):
            return

        in_range = [e for e in candidates
                    if e.is_alive
                    and self._can_attack(attacker, e)
                    and self._distance(attacker, e) <= attacker.range]
        if not in_range:
            return

        target = min(in_range, key=lambda e: self._distance(attacker, e))
        self._damage_unit(target, attacker.damage, attacker.owner)
        attacker.last_attack_tick = self.tick

    def _resolve_combat(self) -> None:
        """Resolve all troop attacks (buildings fire in ``_tower_attacks``)."""
        for unit in list(self.player_units) + list(self.opponent_units):
            if not unit.is_alive or unit.is_building:
                continue
            if unit.status == UnitStatus.STUNNED:
                continue  # Stunned units cannot attack
            enemies = (self.opponent_units if unit.owner == "player"
                       else self.player_units)
            self._perform_attack(unit, list(enemies))

    def _tower_attacks(self) -> None:
        """Process tower attacks."""
        for tower in list(self.player_towers) + list(self.opponent_towers):
            if not tower.is_alive:
                continue

            # Only the king is gated; princess towers are always live.
            if tower.is_king and not tower.is_active:
                continue

            enemies = (self.opponent_units if tower.owner == "player"
                       else self.player_units)
            # Towers shoot troops, not other structures.
            self._perform_attack(tower, [e for e in enemies if not e.is_building])

    def _handle_unit_death(self, unit: UnitState) -> None:
        """Book-keeping for a unit that just died.

        Runs exactly once per death (driven by ``UnitState.just_died``). For
        towers this awards crowns to the destroying side and activates the
        owner's king tower, which previously never happened -- crowns were only
        ever granted for a king kill, so any match that reached time-up
        compared 0 against 0 and was scored a draw.

        Counter convention: ``<side>_towers_destroyed`` counts towers *lost* by
        that side, matching how the evaluator reads them.
        """
        if not unit.is_building:
            return

        if unit.owner == "player":
            self.player_towers_destroyed += 1
            self.opponent_trophies += self._crown_value(unit)
            self._activate_king("player")
        else:
            self.opponent_towers_destroyed += 1
            self.player_trophies += self._crown_value(unit)
            self._activate_king("opponent")

    def _crown_value(self, tower: UnitState) -> int:
        """Crowns awarded for destroying ``tower``."""
        defn = TOWER_DEFS["p1_king" if tower.is_king else "p1_princess_left"]
        return int(getattr(defn, "crown_reward", 1))

    def _activate_king(self, owner: str) -> None:
        """Activate a side's king tower (it wakes when a princess falls)."""
        towers = self.player_towers if owner == "player" else self.opponent_towers
        for tower in towers:
            if tower.is_king and tower.is_alive:
                tower.is_active = True

    def _distance(self, a: UnitState, b: UnitState) -> float:
        """Euclidean distance between two units."""
        return math.sqrt((a.col - b.col) ** 2 + (a.row - b.row) ** 2)

    def _cycle_played_card(self, player: str, hand_idx: int) -> None:
        """Replace a played card with the next card from the deck queue.

        Clash Royale's deck is a fixed rotation: the card you play goes to the
        back of the queue and the front of the queue takes its place in hand.
        The previous implementation instead replaced *all four* slots every
        third tick regardless of what was played, so a policy could never learn
        what a given hand index meant.
        """
        if player == "player":
            hand, queue = self.player_hand, self.player_deck_queue
        else:
            hand, queue = self.opponent_hand, self.opponent_deck_queue

        if not (0 <= hand_idx < len(hand)):
            return
        played = hand[hand_idx]
        if queue:
            hand[hand_idx] = queue.pop(0)
        queue.append(played)

    def _check_win_conditions(self) -> SimulationStepResult:
        """Check if the game has ended and compute rewards.

        Crowns (``*_trophies``) are awarded in ``_handle_unit_death`` as towers
        fall, so the time-up comparison below reflects the whole match rather
        than only king kills.
        """
        player_king = next((t for t in self.player_towers if t.is_king), None)
        opp_king = next((t for t in self.opponent_towers if t.is_king), None)

        if player_king is not None and not player_king.is_alive:
            self.terminated = True
            return SimulationStepResult(
                rewards={"player": -1.0, "opponent": 1.0},
                terminated=True,
                info={"winner": "opponent", "reason": "king_tower_destroyed"},
            )

        if opp_king is not None and not opp_king.is_alive:
            self.terminated = True
            return SimulationStepResult(
                rewards={"player": 1.0, "opponent": -1.0},
                terminated=True,
                info={"winner": "player", "reason": "king_tower_destroyed"},
            )

        # Regulation ended: go to overtime if crowns are level, else finish.
        if not self.is_overtime and self.tick >= self.match_duration_ticks:
            if self.player_trophies == self.opponent_trophies and self.overtime_ticks > 0:
                self.is_overtime = True
                return SimulationStepResult(
                    rewards={"player": 0.0, "opponent": 0.0},
                    terminated=False,
                    truncated=False,
                    info={"phase": "overtime"},
                )
            return self._finish_on_crowns()

        # Overtime expired.
        if self.tick >= self.match_duration_ticks + self.overtime_ticks:
            return self._finish_on_crowns()

        # ── Shaped rewards during play ──────────────────────────────────────
        player_towers_alive = sum(1 for t in self.player_towers if t.is_alive)
        opp_towers_alive = sum(1 for t in self.opponent_towers if t.is_alive)

        player_troops = [u for u in self.player_units
                         if u.is_alive and not u.is_building]
        opp_troops = [u for u in self.opponent_units
                      if u.is_alive and not u.is_building]
        player_units_alive = len(player_troops)
        opp_units_alive = len(opp_troops)

        # Fraction of committed HP still standing, over live troops only.
        player_total_hp = sum(u.hp for u in player_troops)
        opp_total_hp = sum(u.hp for u in opp_troops)
        player_hp_frac = (player_total_hp / max(sum(u.max_hp for u in player_troops), 1.0)
                          if player_troops else 0.0)
        opp_hp_frac = (opp_total_hp / max(sum(u.max_hp for u in opp_troops), 1.0)
                       if opp_troops else 0.0)

        # Lane pressure: how far each side has pushed into enemy territory.
        # (The previous version added each unit to both lanes, which just
        # recomputed the unit count and carried no positional information.)
        player_lane_pressure = sum(
            max(0.0, (self.BRIDGE_ROW - u.row) / max(self.BRIDGE_ROW, 1))
            for u in player_troops)
        opp_lane_pressure = sum(
            max(0.0, (u.row - self.BRIDGE_ROW) /
                max(self.GRID_ROWS - 1 - self.BRIDGE_ROW, 1))
            for u in opp_troops)

        elixir_advantage = (self.player_elixir - self.opponent_elixir) / self.elixir_max
        crown_diff = self.player_trophies - self.opponent_trophies

        # Ramps 0 -> 1 over the last 30% of regulation.
        time_pressure = float(np.clip(
            (self.tick - 0.7 * self.match_duration_ticks) /
            max(0.3 * self.match_duration_ticks, 1.0), 0.0, 1.0))

        def _shaped(crowns, towers_mine, towers_theirs, elixir, hp_frac,
                    mine_alive, theirs_alive, dmg_dealt, dmg_taken,
                    press_mine, press_theirs, elixir_adv):
            return (
                0.30 * crowns / 3.0                                  # Crowns
                + 0.20 * (towers_theirs - towers_mine) / 3.0         # Tower lead
                + 0.15 * (dmg_dealt - dmg_taken) / 10000.0           # Net tower dmg
                + 0.10 * hp_frac                                     # Board health
                + 0.10 * (mine_alive - theirs_alive) / 10.0          # Troop count
                + 0.05 * (press_mine - press_theirs) / 5.0           # Territory
                + 0.05 * elixir / self.elixir_max                    # Elixir banked
                + 0.05 * elixir_adv                                  # Elixir edge
            )

        player_reward = _shaped(
            crown_diff, player_towers_alive, opp_towers_alive,
            self.player_elixir, player_hp_frac, player_units_alive, opp_units_alive,
            self._tower_damage_dealt["opponent"], self._tower_damage_dealt["player"],
            player_lane_pressure, opp_lane_pressure, elixir_advantage)

        opponent_reward = _shaped(
            -crown_diff, opp_towers_alive, player_towers_alive,
            self.opponent_elixir, opp_hp_frac, opp_units_alive, player_units_alive,
            self._tower_damage_dealt["player"], self._tower_damage_dealt["opponent"],
            opp_lane_pressure, player_lane_pressure, -elixir_advantage)

        return SimulationStepResult(
            rewards={"player": player_reward, "opponent": opponent_reward},
            terminated=False,
            truncated=False,
            info={
                "player_towers_alive": player_towers_alive,
                "opponent_towers_alive": opp_towers_alive,
                "player_units_alive": player_units_alive,
                "opponent_units_alive": opp_units_alive,
                "player_elixir": self.player_elixir,
                "opponent_elixir": self.opponent_elixir,
                "player_crowns": self.player_trophies,
                "opponent_crowns": self.opponent_trophies,
                "player_tower_damage": self._tower_damage_dealt["player"],
                "opponent_tower_damage": self._tower_damage_dealt["opponent"],
                "player_lane_pressure": player_lane_pressure,
                "opponent_lane_pressure": opp_lane_pressure,
                "elixir_advantage": elixir_advantage,
                "time_pressure": time_pressure,
            },
        )

    def _finish_on_crowns(self) -> SimulationStepResult:
        """End the match, scoring by crowns earned."""
        self.terminated = True
        if self.player_trophies > self.opponent_trophies:
            winner, rewards = "player", {"player": 1.0, "opponent": -1.0}
        elif self.opponent_trophies > self.player_trophies:
            winner, rewards = "opponent", {"player": -1.0, "opponent": 1.0}
        else:
            winner, rewards = "tie", {"player": 0.0, "opponent": 0.0}
        return SimulationStepResult(
            rewards=rewards,
            terminated=True,
            info={"winner": winner, "reason": "time_up",
                  "player_crowns": self.player_trophies,
                  "opponent_crowns": self.opponent_trophies},
        )

    def _record_replay_frame(self) -> None:
        """Record a replay frame for post-match analysis."""
        frame = ReplayFrame(
            tick=self.tick,
            player_elixir=self.player_elixir,
            opponent_elixir=self.opponent_elixir,
            player_trophies=self.player_trophies,
            opponent_trophies=self.opponent_trophies,
            player_towers_destroyed=self.player_towers_destroyed,
            opponent_towers_destroyed=self.opponent_towers_destroyed,
            is_overtime=self.is_overtime,
            player_hand=list(self.player_hand),
            opponent_hand=list(self.opponent_hand),
        )

        # Record unit positions
        for u in self.player_units:
            if u.is_alive:
                frame.player_units.append({
                    "type": u.unit_type,
                    "col": u.col,
                    "row": u.row,
                    "hp": u.hp,
                    "max_hp": u.max_hp,
                })
        for u in self.opponent_units:
            if u.is_alive:
                frame.opponent_units.append({
                    "type": u.unit_type,
                    "col": u.col,
                    "row": u.row,
                    "hp": u.hp,
                    "max_hp": u.max_hp,
                })

        # Record tower states
        for t in self.player_towers:
            frame.player_towers.append({
                "type": t.unit_type,
                "col": t.col,
                "row": t.row,
                "hp": t.hp,
                "max_hp": t.max_hp,
                "is_alive": t.is_alive,
                "is_active": t.is_active if hasattr(t, 'is_active') else False,
            })
        for t in self.opponent_towers:
            frame.opponent_towers.append({
                "type": t.unit_type,
                "col": t.col,
                "row": t.row,
                "hp": t.hp,
                "max_hp": t.max_hp,
                "is_alive": t.is_alive,
                "is_active": t.is_active if hasattr(t, 'is_active') else False,
            })

        # Record arena state
        frame.arena_state = self.arena.copy()

        self.replay_frames.append(frame)

    def get_replay_data(self) -> List[ReplayFrame]:
        """Get the full replay data for this match."""
        return self.replay_frames

    def get_match_summary(self) -> Dict:
        """Get a summary of the match results."""
        return {
            "duration_ticks": self.tick,
            "player_trophies": self.player_trophies,
            "opponent_trophies": self.opponent_trophies,
            "player_towers_destroyed": self.player_towers_destroyed,
            "opponent_towers_destroyed": self.opponent_towers_destroyed,
            "winner": ("player" if self.player_trophies > self.opponent_trophies
                      else "opponent" if self.opponent_trophies > self.player_trophies
                      else "tie"),
            "is_overtime": self.is_overtime,
            "player_tower_damage": self._tower_damage_dealt["player"],
            "opponent_tower_damage": self._tower_damage_dealt["opponent"],
            # Cumulative: corpses are reclaimed each tick, so these cannot be
            # recovered by scanning the live unit lists.
            "player_units_spawned": self._units_spawned["player"],
            "opponent_units_spawned": self._units_spawned["opponent"],
            "action_count": len(self.action_history),
        }