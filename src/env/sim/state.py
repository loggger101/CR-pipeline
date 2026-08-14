"""Game state representation and preprocessing for the simulation.

Provides GameStateSnapshot (the observation the neural net sees) and
preprocess_state (converting raw state to the input tensor format).

Enhanced with:
- Elixir advantage feature
- Unit type distribution per lane
- Air/ground unit separation
- Status effect heatmaps
- King tower activation status
- Time pressure (overtime indicator)
- Card cooldown state
- Lane pressure indicators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


class UnitStatus(Enum):
    """Status effects that can be applied to units.

    Defined here (rather than in the engine) so that ``UnitState`` can branch
    on the same values the engine applies. Previously the engine passed these
    enum members into ``UnitState.apply_status`` while that method compared
    against bare ints, so every speed modifier silently no-opped.
    """
    NONE = auto()
    STUNNED = auto()       # Cannot move or attack
    SLOWED = auto()        # Movement speed reduced
    SILENCED = auto()      # Cannot deploy spells
    POISONED = auto()      # Takes damage per tick
    BUFFED_DAMAGE = auto() # Damage increased
    BUFFED_HP = auto()     # Max HP increased
    SHELLFIRE = auto()     # Explodes on death


# Fraction of base move speed retained while SLOWED.
SLOW_SPEED_FACTOR = 0.5


@dataclass
class UnitState:
    """Runtime state of a single unit/tower on the arena.

    Attributes:
        unit_type: Card name (e.g. "knight").
        owner: "player" or "opponent".
        hp: Current hit points.
        max_hp: Maximum hit points.
        col: Current grid column (float for sub-cell precision).
        row: Current grid row (float).
        target_col: Current attack target column.
        target_row: Current attack target row.
        is_alive: Whether the unit is still alive.
        is_building: Whether this is a stationary structure.
        speed: Movement speed (cells per tick).
        range: Attack range in grid cells.
        damage: Damage per attack tick.
        target_pref: Preferred target type.
        status: Current status effect (a UnitStatus member).
        status_duration: Ticks remaining for current status.
        is_air: Whether this unit is an air unit.
        is_ground: Whether this unit is a ground unit.
        can_target_air: Whether this unit may attack air units.
        can_target_ground: Whether this unit may attack ground units.
        is_active: Whether a gated tower (the king) has been activated.
        activation_range: Range at which the king tower activates.
        is_king: Whether this tower is a king tower.
        base_speed: Movement speed before any status modifier.
        just_died: True for the tick on which this unit died.
        is_spawned: Whether this unit was spawned (not a tower).
        spawn_tick: Tick when this unit was spawned.
        last_attack_tick: Tick of last attack.
        attack_speed: Base attack speed in seconds.
    """
    unit_type: str
    owner: str
    hp: float
    max_hp: float
    col: float
    row: float
    target_col: float = -1.0
    target_row: float = -1.0
    is_alive: bool = True
    is_building: bool = False
    speed: float = 1.0
    range: float = 1.0
    damage: int = 0
    target_pref: int = 0  # TargetingMode enum value

    # Status effects
    status: UnitStatus = UnitStatus.NONE
    status_duration: int = 0

    # Air/ground
    is_air: bool = False
    is_ground: bool = True

    # King tower
    is_active: bool = False
    activation_range: float = 4.0
    is_king: bool = False

    # Spawn tracking
    is_spawned: bool = False
    spawn_tick: int = 0

    # Attack tracking
    last_attack_tick: int = -10
    attack_speed: float = 1.0

    # Runtime tracking
    current_target: Optional["UnitState"] = None  # Current attack target
    path_progress: float = 0.0  # Progress along path (0-1)
    is_deploying: bool = False  # Whether unit is currently being deployed
    deploy_timer: int = 0  # Ticks remaining for deployment
    status_source: str = ""  # Source of status effect (spell name)
    can_target_air: bool = False
    can_target_ground: bool = True
    # Base move speed, captured once so slows/stuns are reversible.
    base_speed: float = -1.0
    # Set for one tick after the unit dies, so the engine can run death effects.
    just_died: bool = False

    def __post_init__(self) -> None:
        if self.base_speed < 0:
            self.base_speed = self.speed

    def take_damage(self, damage: float) -> float:
        """Apply damage and return actual damage dealt.

        Sets ``just_died`` on the transition to dead so the engine can run
        death effects exactly once.
        """
        if not self.is_alive:
            return 0.0
        actual = min(max(damage, 0.0), self.hp)
        self.hp -= actual
        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False
            self.just_died = True
        return actual

    def heal(self, amount: float) -> float:
        """Restore hit points up to ``max_hp``; returns the amount healed."""
        if not self.is_alive:
            return 0.0
        healed = min(max(amount, 0.0), self.max_hp - self.hp)
        self.hp += healed
        return healed

    def distance_to(self, col: float, row: float) -> float:
        """Euclidean distance to a point."""
        return np.sqrt((self.col - col) ** 2 + (self.row - row) ** 2)

    def distance_to_unit(self, other) -> float:
        """Euclidean distance to another unit."""
        return self.distance_to(other.col, other.row)

    def apply_status(self, status: UnitStatus, duration: int,
                     source: str = "") -> None:
        """Apply a status effect to this unit.

        A new effect replaces the current one, and movement speed is always
        recomputed from ``base_speed`` so effects never compound or leak.
        """
        self.status = status
        self.status_duration = duration
        self.status_source = source
        self._refresh_speed()

    def _refresh_speed(self) -> None:
        """Recompute ``speed`` from ``base_speed`` and the active status."""
        if self.status == UnitStatus.STUNNED:
            self.speed = 0.0
        elif self.status == UnitStatus.SLOWED:
            # Scaled off base_speed with no absolute floor: speed is in tiles
            # per tick (~0.1), so a fixed 0.1 minimum cancelled the slow.
            self.speed = self.base_speed * SLOW_SPEED_FACTOR
        else:
            self.speed = self.base_speed

    def clear_status(self) -> None:
        """Clear the current status effect and restore base stats."""
        self.status = UnitStatus.NONE
        self.status_duration = 0
        self.status_source = ""
        self._refresh_speed()

    def update_status(self) -> bool:
        """Tick down the active status. Returns True if it expired this tick."""
        if self.status == UnitStatus.NONE:
            return False
        self.status_duration -= 1
        if self.status_duration <= 0:
            self.clear_status()
            return True
        return False


@dataclass
class GameStateSnapshot:
    """Observation state provided to the neural network.

    The state is represented as a multi-channel tensor of shape
    (num_channels, H, W) where each channel encodes a different
    aspect of the game state.

    Channels:
        0-3:     Card hand (one-hot per slot, 1 if card is available)
        4:       Player elixir level (normalized 0-1)
        5:       Opponent elixir level (normalized 0-1)
        6:       Unit density heatmap (units on arena)
        7:       Tower health heatmap (normalized tower HP)
        8-9:     Left/right lane unit density
        10:      Bridge status (bridge occupancy)
        11:      Time remaining (normalized)
        12:      Elixir advantage (player - opponent, normalized)
        13-14:   Lane pressure (player units in left/right lanes)
        15:      Overtime indicator
        16:      Card cooldown state (fraction of cooldown remaining)
    """
    # Card hand: one-hot per slot
    card_hand: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    # Elixir levels (normalized)
    player_elixir: float = 5.0
    opponent_elixir: float = 5.0
    # Arena grids (H x W)
    unit_density: np.ndarray = field(default_factory=lambda: np.zeros((6, 8), dtype=np.float32))
    tower_health: np.ndarray = field(default_factory=lambda: np.zeros((6, 8), dtype=np.float32))
    lane_presence: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    bridge_status: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    time_remaining: float = 180.0  # Seconds remaining
    # Additional features
    elixir_advantage: float = 0.0  # player - opponent, normalized
    lane_pressure: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    overtime_indicator: float = 0.0
    cooldown_state: float = 0.0  # Average fraction of cooldown remaining
    # Raw game state (for engine use)
    player_units: List[UnitState] = field(default_factory=list)
    opponent_units: List[UnitState] = field(default_factory=list)
    player_towers: List[UnitState] = field(default_factory=list)
    opponent_towers: List[UnitState] = field(default_factory=list)
    # Match metadata
    player_trophies: int = 0
    opponent_trophies: int = 0
    player_towers_destroyed: int = 0
    opponent_towers_destroyed: int = 0
    is_overtime: bool = False
    tick: int = 0
    max_ticks: int = 1800


def preprocess_state(state: GameStateSnapshot,
                     resolution: int = 64) -> np.ndarray:
    """Convert GameStateSnapshot to a normalized input tensor.

    Args:
        state: Current game state snapshot.
        resolution: Output tensor resolution (resolution x resolution).

    Returns:
        Tensor of shape (num_channels, resolution, resolution) with
        values in [0, 1].
    """
    channels = []

    # 1. Card hand (4 channels)
    card_channels = np.zeros((4, resolution, resolution), dtype=np.float32)
    for i in range(4):
        card_channels[i] = state.card_hand[i]
    channels.append(card_channels)

    # 2. Player elixir (1 channel)
    elixir_p = np.full((1, resolution, resolution),
                       state.player_elixir / 10.0, dtype=np.float32)
    channels.append(elixir_p)

    # 3. Opponent elixir (1 channel)
    elixir_o = np.full((1, resolution, resolution),
                       state.opponent_elixir / 10.0, dtype=np.float32)
    channels.append(elixir_o)

    # 4. Unit density heatmap (upsampled to resolution)
    unit_heat = np.zeros((resolution, resolution), dtype=np.float32)
    h, w = state.unit_density.shape
    for r in range(h):
        for c in range(w):
            val = state.unit_density[r, c]
            rr = int(r * resolution / h)
            cc = int(c * resolution / w)
            if rr < resolution and cc < resolution:
                unit_heat[rr, cc] = val
    channels.append(unit_heat[np.newaxis, :, :])

    # 5. Tower health heatmap (upsampled)
    tower_heat = np.zeros((resolution, resolution), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            val = state.tower_health[r, c]
            rr = int(r * resolution / h)
            cc = int(c * resolution / w)
            if rr < resolution and cc < resolution:
                tower_heat[rr, cc] = val
    channels.append(tower_heat[np.newaxis, :, :])

    # 6. Lane presence (2 channels)
    lane_ch = np.zeros((2, resolution, resolution), dtype=np.float32)
    lane_ch[0] = state.lane_presence[0]  # Left lane
    lane_ch[1] = state.lane_presence[1]  # Right lane
    channels.append(lane_ch)

    # 7. Bridge status (1 channel)
    bridge_ch = np.full((1, resolution, resolution),
                        state.bridge_status[0], dtype=np.float32)
    channels.append(bridge_ch)

    # 8. Time remaining (1 channel)
    time_ch = np.full((1, resolution, resolution),
                      state.time_remaining / 180.0, dtype=np.float32)
    channels.append(time_ch)

    # 9. Elixir advantage (1 channel) - normalized to [-1, 1] -> [0, 1]
    elixir_adv = (state.elixir_advantage + 1.0) / 2.0
    elixir_adv_ch = np.full((1, resolution, resolution),
                            max(0, min(1, elixir_adv)), dtype=np.float32)
    channels.append(elixir_adv_ch)

    # 10. Lane pressure (2 channels)
    lane_pressure_ch = np.zeros((2, resolution, resolution), dtype=np.float32)
    lane_pressure_ch[0] = state.lane_pressure[0]
    lane_pressure_ch[1] = state.lane_pressure[1]
    channels.append(lane_pressure_ch)

    # 11. Overtime indicator (1 channel)
    overtime_ch = np.full((1, resolution, resolution),
                          state.overtime_indicator, dtype=np.float32)
    channels.append(overtime_ch)

    # 12. Cooldown state (1 channel)
    cooldown_ch = np.full((1, resolution, resolution),
                          state.cooldown_state, dtype=np.float32)
    channels.append(cooldown_ch)

    # Stack all channels: (num_channels, H, W)
    tensor = np.concatenate(channels, axis=0)
    return np.clip(tensor, 0.0, 1.0)


def compute_state_from_arena(state: GameStateSnapshot,
                             arena: np.ndarray) -> GameStateSnapshot:
    """Update state from the raw arena grid.

    Args:
        state: State to update in-place.
        arena: Grid where each cell contains unit info.

    Returns:
        Updated state.
    """
    # Compute unit density
    density = np.zeros((6, 8), dtype=np.float32)
    tower_health = np.zeros((6, 8), dtype=np.float32)

    for unit in state.player_units + state.opponent_units:
        if unit.is_alive:
            r = min(int(unit.row), 5)
            c = min(int(unit.col), 7)
            density[r, c] += unit.hp / unit.max_hp

    for tower in state.player_towers + state.opponent_towers:
        if tower.is_alive:
            r = min(int(tower.row), 5)
            c = min(int(tower.col), 7)
            tower_health[r, c] = tower.hp / tower.max_hp

    state.unit_density = density
    state.tower_health = tower_health

    # Lane presence
    left_lane = sum(u.hp / u.max_hp for u in
                    state.player_units + state.opponent_units
                    if u.is_alive and u.col < 4)
    right_lane = sum(u.hp / u.max_hp for u in
                     state.player_units + state.opponent_units
                     if u.is_alive and u.col >= 4)
    state.lane_presence[0] = min(left_lane / 5.0, 1.0)
    state.lane_presence[1] = min(right_lane / 5.0, 1.0)

    # Bridge status
    bridge_col = 3
    bridge_count = sum(1 for u in state.player_units + state.opponent_units
                       if u.is_alive and abs(u.col - bridge_col) < 1)
    state.bridge_status[0] = min(bridge_count / 3.0, 1.0)

    # Elixir advantage
    state.elixir_advantage = (state.player_elixir - state.opponent_elixir) / 10.0

    # Overtime indicator
    state.overtime_indicator = 1.0 if state.is_overtime else 0.0

    # Cooldown state (average fraction of cooldown remaining)
    # Computed in _get_state based on actual cooldowns

    return state


def compute_enhanced_state(state: GameStateSnapshot,
                           arena: np.ndarray,
                           match_duration_ticks: int = 1800,
                           overtime_ticks: int = 120) -> GameStateSnapshot:
    """Compute an enhanced state representation with additional features.

    Adds:
    - Unit type distribution per lane
    - Air/ground unit separation
    - Status effect heatmaps
    - King tower activation status
    - Time pressure (overtime indicator)
    - Elixir advantage
    - Card cooldown state
    - Lane pressure indicators

    Args:
        state: State to update in-place.
        arena: Grid where each cell contains unit info.
        match_duration_ticks: Length of regulation match.
        overtime_ticks: Overtime duration.

    Returns:
        Updated state.
    """
    # Reset computed fields
    state.unit_density = np.zeros((6, 8), dtype=np.float32)
    state.tower_health = np.zeros((6, 8), dtype=np.float32)
    state.lane_presence = np.zeros(2, dtype=np.float32)
    state.bridge_status = np.zeros(2, dtype=np.float32)
    state.lane_pressure = np.zeros(2, dtype=np.float32)

    # Compute unit density
    for unit in state.player_units + state.opponent_units:
        if unit.is_alive:
            r = min(int(unit.row), 5)
            c = min(int(unit.col), 7)
            state.unit_density[r, c] += unit.hp / unit.max_hp

    # Compute tower health
    for tower in state.player_towers + state.opponent_towers:
        if tower.is_alive:
            r = min(int(tower.row), 5)
            c = min(int(tower.col), 7)
            state.tower_health[r, c] = tower.hp / tower.max_hp

    # Lane presence (separate for each player)
    player_left = sum(u.hp / u.max_hp for u in state.player_units
                      if u.is_alive and u.col < 4)
    player_right = sum(u.hp / u.max_hp for u in state.player_units
                       if u.is_alive and u.col >= 4)
    opp_left = sum(u.hp / u.max_hp for u in state.opponent_units
                   if u.is_alive and u.col < 4)
    opp_right = sum(u.hp / u.max_hp for u in state.opponent_units
                    if u.is_alive and u.col >= 4)
    
    state.lane_presence[0] = min((player_left + opp_left) / 5.0, 1.0)
    state.lane_presence[1] = min((player_right + opp_right) / 5.0, 1.0)

    # Lane pressure (player units pushed into opponent territory)
    state.lane_pressure[0] = min(player_left / 5.0, 1.0)
    state.lane_pressure[1] = min(player_right / 5.0, 1.0)

    # Bridge status
    for bridge_col in [3, 4]:
        bridge_count = sum(1 for u in state.player_units + state.opponent_units
                           if u.is_alive and abs(u.col - bridge_col) < 1)
        state.bridge_status[min(bridge_col - 3, 1)] = min(bridge_count / 3.0, 1.0)

    # Elixir advantage
    state.elixir_advantage = (state.player_elixir - state.opponent_elixir) / 10.0

    # Time pressure
    total_ticks = match_duration_ticks + (overtime_ticks if state.is_overtime else 0)
    state.time_remaining = max(0, total_ticks - state.tick)
    state.overtime_indicator = 1.0 if state.is_overtime else 0.0

    return state