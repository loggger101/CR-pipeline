"""Game state representation and preprocessing for the simulation.

Provides GameStateSnapshot (the observation the neural net sees) and
preprocess_state (converting raw state to the input tensor format).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
    target_pref: int = 0  # TargetPreference enum value

    def take_damage(self, damage: float) -> float:
        """Apply damage and return actual damage dealt."""
        actual = min(damage, self.hp)
        self.hp -= actual
        if self.hp <= 0:
            self.hp = 0
            self.is_alive = False
        return actual

    def distance_to(self, col: float, row: float) -> float:
        """Euclidean distance to a point."""
        return np.sqrt((self.col - col) ** 2 + (self.row - row) ** 2)


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
            # Fill corresponding region in upscaled grid
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

    return state
