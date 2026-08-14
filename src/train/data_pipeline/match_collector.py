"""Match data collection for training.

Collects and stores:
- Full match state snapshots
- Action history
- Reward signals
- Game outcome data
- Agent predictions
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatchCollectorConfig:
    """Configuration for match data collection."""
    enabled: bool = True
    save_replays: bool = True
    save_state_snapshots: bool = True
    snapshot_interval: int = 10  # Save every N ticks
    max_replays: int = 1000
    output_dir: str = "runs/matches"


@dataclass
class MatchData:
    """Complete match data.

    Attributes:
        match_id: Unique match identifier.
        generation: Training generation.
        agent1_id: First agent ID.
        agent2_id: Second agent ID.
        agent1_side: Side for agent 1 (left/right).
        agent2_side: Side for agent 2 (left/right).
        duration_ticks: Match duration.
        winner: Winner agent ID.
        agent1_trophies: Agent 1 final trophy count.
        agent2_trophies: Agent 2 final trophy count.
        agent1_towers: Towers destroyed by agent 1.
        agent2_towers: Towers destroyed by agent 2.
        state_snapshots: List of state snapshots.
        action_history: List of actions taken.
        reward_history: Per-tick rewards.
        metadata: Additional match metadata.
    """
    match_id: str
    generation: int
    agent1_id: str
    agent2_id: str
    agent1_side: str = "left"
    agent2_side: str = "right"
    duration_ticks: int = 0
    winner: Optional[str] = None
    agent1_trophies: int = 0
    agent2_trophies: int = 0
    agent1_towers: int = 0
    agent2_towers: int = 0
    state_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    reward_history: List[Dict[str, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "generation": self.generation,
            "agent1_id": self.agent1_id,
            "agent2_id": self.agent2_id,
            "agent1_side": self.agent1_side,
            "agent2_side": self.agent2_side,
            "duration_ticks": self.duration_ticks,
            "winner": self.winner,
            "agent1_trophies": self.agent1_trophies,
            "agent2_trophies": self.agent2_trophies,
            "agent1_towers": self.agent1_towers,
            "agent2_towers": self.agent2_towers,
            "state_snapshots_count": len(self.state_snapshots),
            "action_history_count": len(self.action_history),
            "reward_history_count": len(self.reward_history),
            "metadata": self.metadata,
        }


class MatchCollector:
    """Collects and stores match data for training analysis."""

    def __init__(self, config: Optional[MatchCollectorConfig] = None):
        self.config = config or MatchCollectorConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.match_count: int = 0

    def collect_match(
        self,
        generation: int,
        agent1_id: str,
        agent2_id: str,
        state_snapshots: List[Dict[str, Any]],
        action_history: List[Dict[str, Any]],
        reward_history: List[Dict[str, float]],
        winner: Optional[str] = None,
        agent1_trophies: int = 0,
        agent2_trophies: int = 0,
        agent1_towers: int = 0,
        agent2_towers: int = 0,
        duration_ticks: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MatchData:
        """Collect and store match data.

        Args:
            generation: Training generation.
            agent1_id: First agent ID.
            agent2_id: Second agent ID.
            state_snapshots: State snapshots during match.
            action_history: Actions taken during match.
            reward_history: Per-tick rewards.
            winner: Winning agent ID.
            agent1_trophies: Agent 1 final trophy count.
            agent2_trophies: Agent 2 final trophy count.
            agent1_towers: Towers destroyed by agent 1.
            agent2_towers: Towers destroyed by agent 2.
            duration_ticks: Match duration.
            metadata: Additional metadata.

        Returns:
            Collected MatchData.
        """
        match_id = f"gen{generation:04d}_agent1_{agent1_id}_agent2_{agent2_id}"
        match_data = MatchData(
            match_id=match_id,
            generation=generation,
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            winner=winner,
            agent1_trophies=agent1_trophies,
            agent2_trophies=agent2_trophies,
            agent1_towers=agent1_towers,
            agent2_towers=agent2_towers,
            duration_ticks=duration_ticks,
            state_snapshots=state_snapshots[-100:] if state_snapshots else [],
            action_history=action_history[-500:] if action_history else [],
            reward_history=reward_history[-1000:] if reward_history else [],
            metadata=metadata or {},
        )

        # Save match data
        gen_dir = self.output_dir / f"gen_{generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save match summary
        summary_path = gen_dir / f"{match_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(match_data.to_dict(), f, indent=2)

        # Save replay data
        if self.config.save_replays and (state_snapshots or action_history):
            replay_path = gen_dir / f"{match_id}_replay.json"
            replay_data = {
                "match_id": match_id,
                "snapshots": state_snapshots[-50:] if state_snapshots else [],
                "actions": action_history[-200:] if action_history else [],
            }
            with open(replay_path, "w") as f:
                json.dump(replay_data, f, indent=2)

        self.match_count += 1
        logger.debug(f"Collected match {match_id} ({self.match_count} total)")
        return match_data

    def get_match_count(self, generation: Optional[int] = None) -> int:
        """Get number of collected matches."""
        if generation is None:
            return self.match_count
        gen_dir = self.output_dir / f"gen_{generation:04d}"
        if gen_dir.exists():
            return len([f for f in gen_dir.glob("*_summary.json")])
        return 0
