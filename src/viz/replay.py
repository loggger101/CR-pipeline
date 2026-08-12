"""Replay viewer for simulation matches.

Provides:
- Replay data loading from action history
- Step-by-step replay playback
- Frame rendering of game state
- Action visualization overlay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..env.sim.engine import SimulationEngine
from ..env.sim.state import GameStateSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ReplayFrame:
    """A single frame in a replay.

    Attributes:
        tick: Game tick number.
        state: Game state snapshot at this tick.
        action: Action taken by the agent.
        reward: Reward received.
        arena: Arena grid state.
        player_units: Player unit positions.
        opponent_units: Opponent unit positions.
    """
    tick: int
    state: Optional[GameStateSnapshot] = None
    action: Optional[dict] = None
    reward: float = 0.0
    arena: Optional[np.ndarray] = None
    player_units: List[Tuple[float, float, str]] = field(default_factory=list)
    opponent_units: List[Tuple[float, float, str]] = field(default_factory=list)
    player_trophies: int = 0
    opponent_trophies: int = 0
    player_elixir: float = 5.0
    opponent_elixir: float = 5.0


class ReplayViewer:
    """Loads and plays back simulation match replays.

    Stores the full tick-by-tick state of a match for visualization.
    Supports step-through playback and summary statistics.
    """

    def __init__(self):
        self.frames: List[ReplayFrame] = []
        self.metadata: Dict = {}
        self.current_frame: int = 0

    def load_replay(self, engine: SimulationEngine) -> None:
        """Load replay data from a simulation engine.

        Args:
            engine: SimulationEngine with a completed match.
        """
        self.frames = []
        self.metadata = {
            "match_duration": engine.tick,
            "player_trophies": engine.player_trophies,
            "opponent_trophies": engine.opponent_trophies,
            "player_towers_destroyed": engine.opponent_towers_destroyed,
            "opponent_towers_destroyed": engine.player_towers_destroyed,
            "winner": "player" if engine.player_trophies > engine.opponent_trophies
                      else "opponent" if engine.opponent_trophies > engine.player_trophies
                      else "tie",
            "is_overtime": engine.is_overtime,
        }

        # Reconstruct frames from action history
        for action_entry in engine.action_history:
            frame = ReplayFrame(
                tick=action_entry["tick"],
                action=action_entry,
            )
            self.frames.append(frame)

        # Add final frame
        final_frame = ReplayFrame(
            tick=engine.tick,
            state=engine._get_state(),
            arena=engine.arena.copy(),
            player_units=[(u.col, u.row, u.unit_type)
                         for u in engine.player_units if u.is_alive],
            opponent_units=[(u.col, u.row, u.unit_type)
                           for u in engine.opponent_units if u.is_alive],
            player_trophies=engine.player_trophies,
            opponent_trophies=engine.opponent_trophies,
            player_elixir=engine.player_elixir,
            opponent_elixir=engine.opponent_elixir,
        )
        self.frames.append(final_frame)

        logger.info(f"Loaded replay: {len(self.frames)} frames, "
                    f"duration={self.metadata['match_duration']} ticks")

    def get_frame(self, index: int) -> Optional[ReplayFrame]:
        """Get a specific frame by index.

        Args:
            index: Frame index.

        Returns:
            ReplayFrame at the given index, or None.
        """
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def get_summary(self) -> Dict:
        """Get replay summary statistics.

        Returns:
            Dictionary of summary data.
        """
        if not self.frames:
            return {}

        player_actions = sum(1 for f in self.frames
                            if f.action and f.action.get("player") == "player")
        opponent_actions = sum(1 for f in self.frames
                              if f.action and f.action.get("player") == "opponent")

        # Card usage statistics
        card_usage: Dict[str, int] = {}
        for f in self.frames:
            if f.action and f.action.get("card"):
                card = f.action["card"]
                card_usage[card] = card_usage.get(card, 0) + 1

        return {
            **self.metadata,
            "total_frames": len(self.frames),
            "player_actions": player_actions,
            "opponent_actions": opponent_actions,
            "card_usage": card_usage,
        }

    def render_arena(self, frame: Optional[ReplayFrame] = None) -> np.ndarray:
        """Render the arena state as a numpy array for display.

        Args:
            frame: ReplayFrame to render. Uses current frame if None.

        Returns:
            Arena visualization array of shape (H, W, 3) with RGB values.
        """
        if frame is None:
            frame = self.frames[self.current_frame] if self.frames else None

        if frame is None or frame.arena is None:
            return np.zeros((6, 8, 3), dtype=np.uint8)

        # Create visualization
        vis = np.ones((6, 8, 3), dtype=np.uint8) * 240  # Light gray background

        # Draw arena grid
        for r in range(6):
            for c in range(8):
                if frame.arena[r, c] != 0:
                    vis[r, c] = [180, 160, 140]  # Occupied cell

        # Draw bridges
        for bridge_col in [3, 4]:
            for r in range(2, 4):
                vis[r, bridge_col] = [100, 150, 200]  # Blue bridge

        # Draw player units (green)
        for col, row, unit_type in frame.player_units:
            r, c = int(row), int(col)
            if 0 <= r < 6 and 0 <= c < 8:
                vis[r, c] = [50, 200, 50]  # Green

        # Draw opponent units (red)
        for col, row, unit_type in frame.opponent_units:
            r, c = int(row), int(col)
            if 0 <= r < 6 and 0 <= c < 8:
                vis[r, c] = [200, 50, 50]  # Red

        return vis

    def next_frame(self) -> Optional[ReplayFrame]:
        """Advance to the next frame.

        Returns:
            Next ReplayFrame, or None if at end.
        """
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            return self.frames[self.current_frame]
        return None

    def prev_frame(self) -> Optional[ReplayFrame]:
        """Go to the previous frame.

        Returns:
            Previous ReplayFrame, or None if at start.
        """
        if self.current_frame > 0:
            self.current_frame -= 1
            return self.frames[self.current_frame]
        return None

    def jump_to_frame(self, index: int) -> Optional[ReplayFrame]:
        """Jump to a specific frame.

        Args:
            index: Frame index to jump to.

        Returns:
            ReplayFrame at the given index.
        """
        self.current_frame = max(0, min(index, len(self.frames) - 1))
        return self.frames[self.current_frame]

    def __len__(self) -> int:
        return len(self.frames)
