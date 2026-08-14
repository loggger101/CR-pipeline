"""Tournament result collection for training analysis.

Collects and stores:
- Full tournament match results
- ELO rating progression
- Head-to-head records
- Bracket data
- Agent performance statistics
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
class TournamentCollectorConfig:
    """Configuration for tournament data collection."""
    enabled: bool = True
    save_bracket_data: bool = True
    save_elo_history: bool = True
    save_h2h_records: bool = True
    max_matches: int = 10000
    output_dir: str = "runs/tournaments"


@dataclass
class TournamentMatchData:
    """Complete tournament match data.

    Attributes:
        match_id: Unique match identifier.
        tournament_id: Parent tournament ID.
        round_number: Round number within tournament.
        agent1_id: First agent ID.
        agent2_id: Second agent ID.
        agent1_elo_before: Agent 1 ELO before match.
        agent2_elo_before: Agent 2 ELO before match.
        agent1_elo_after: Agent 1 ELO after match.
        agent2_elo_after: Agent 2 ELO after match.
        winner: Winning agent ID.
        agent1_trophies: Agent 1 final trophy count.
        agent2_trophies: Agent 2 final trophy count.
        duration_ticks: Match duration.
        metadata: Additional match metadata.
    """
    match_id: str
    tournament_id: str
    round_number: int
    agent1_id: str
    agent2_id: str
    agent1_elo_before: float = 1500.0
    agent2_elo_before: float = 1500.0
    agent1_elo_after: float = 1500.0
    agent2_elo_after: float = 1500.0
    winner: Optional[str] = None
    agent1_trophies: int = 0
    agent2_trophies: int = 0
    duration_ticks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "tournament_id": self.tournament_id,
            "round_number": self.round_number,
            "agent1_id": self.agent1_id,
            "agent2_id": self.agent2_id,
            "agent1_elo_before": self.agent1_elo_before,
            "agent2_elo_before": self.agent2_elo_before,
            "agent1_elo_after": self.agent1_elo_after,
            "agent2_elo_after": self.agent2_elo_after,
            "winner": self.winner,
            "agent1_trophies": self.agent1_trophies,
            "agent2_trophies": self.agent2_trophies,
            "duration_ticks": self.duration_ticks,
            "metadata": self.metadata,
        }


class TournamentCollector:
    """Collects and stores tournament data for analysis."""

    def __init__(self, config: Optional[TournamentCollectorConfig] = None):
        self.config = config or TournamentCollectorConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tournament_count: int = 0
        self.match_count: int = 0
        self._elo_history: Dict[str, List[float]] = {}

    def collect_tournament(
        self,
        tournament_id: str,
        matches: List[Dict[str, Any]],
        elo_history: Dict[str, List[float]],
        bracket_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect and store tournament data.

        Args:
            tournament_id: Unique tournament identifier.
            matches: List of match result dictionaries.
            elo_history: ELO rating progression per agent.
            bracket_data: Bracket visualization data.

        Returns:
            Summary of collected data.
        """
        # Save tournament summary
        tournament_dir = self.output_dir / tournament_id
        tournament_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "tournament_id": tournament_id,
            "match_count": len(matches),
            "agent_count": len(set(
                m.get("agent1_id") for m in matches
            ) | set(m.get("agent2_id") for m in matches)),
            "collected_at": time.time(),
        }

        # Save matches
        matches_path = tournament_dir / "matches.json"
        with open(matches_path, "w") as f:
            json.dump([m for m in matches], f, indent=2)

        # Save ELO history
        if self.config.save_elo_history:
            elo_path = tournament_dir / "elo_history.json"
            with open(elo_path, "w") as f:
                json.dump(elo_history, f, indent=2)
            self._elo_history[tournament_id] = elo_history

        # Save bracket data
        if self.config.save_bracket_data and bracket_data:
            bracket_path = tournament_dir / "bracket.json"
            with open(bracket_path, "w") as f:
                json.dump(bracket_data, f, indent=2)

        # Save H2H records
        if self.config.save_h2h_records:
            h2h = self._compute_h2h_records(matches)
            h2h_path = tournament_dir / "h2h_records.json"
            with open(h2h_path, "w") as f:
                json.dump(h2h, f, indent=2)

        self.tournament_count += 1
        self.match_count += len(matches)
        logger.info(f"Collected tournament {tournament_id} ({len(matches)} matches)")

        return summary

    def _compute_h2h_records(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute head-to-head records from matches."""
        h2h = {}
        for match in matches:
            a1 = match.get("agent1_id", "")
            a2 = match.get("agent2_id", "")
            winner = match.get("winner", "")
            key = tuple(sorted([a1, a2]))

            if key not in h2h:
                h2h[key] = {
                    "agent1": a1,
                    "agent2": a2,
                    "agent1_wins": 0,
                    "agent2_wins": 0,
                    "total_matches": 0,
                }

            h2h[key]["total_matches"] += 1
            if winner == a1:
                h2h[key]["agent1_wins"] += 1
            elif winner == a2:
                h2h[key]["agent2_wins"] += 1

        return h2h

    def get_elo_history(self, tournament_id: str) -> Optional[Dict[str, List[float]]]:
        """Get ELO history for a tournament."""
        return self._elo_history.get(tournament_id)

    def get_tournament_count(self) -> int:
        """Get number of collected tournaments."""
        return self.tournament_count


class TournamentDataset:
    """Manages tournament datasets for analysis."""

    def __init__(self, dataset_dir: str = "runs/tournament_datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.versions: List[TournamentDatasetVersion] = []

    def create_dataset(
        self,
        dataset_id: str,
        tournament_ids: List[str],
        description: str = "",
    ) -> TournamentDatasetVersion:
        """Create a tournament dataset from collected tournaments.

        Args:
            dataset_id: Unique dataset identifier.
            tournament_ids: Source tournament IDs.
            description: Dataset description.

        Returns:
            Created dataset version.
        """
        match_count = 0
        for tid in tournament_ids:
            matches_path = self.output_dir / tid / "matches.json"
            if matches_path.exists():
                with open(matches_path) as f:
                    match_count += len(json.load(f))

        version = TournamentDatasetVersion(
            dataset_id=dataset_id,
            version=f"v{len(self.versions) + 1}",
            tournament_ids=tournament_ids,
            match_count=match_count,
            description=description,
            created_at=time.time(),
        )
        self.versions.append(version)

        # Save version info
        version_path = self.dataset_dir / f"{dataset_id}.json"
        with open(version_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        logger.info(f"Created tournament dataset {dataset_id} ({match_count} matches)")
        return version

    @property
    def output_dir(self) -> Path:
        return self.dataset_dir

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all tournament datasets."""
        datasets = []
        for version in self.versions:
            datasets.append(version.to_dict())
        return datasets


@dataclass
class TournamentDatasetVersion:
    """Version of a tournament dataset."""
    dataset_id: str
    version: str
    tournament_ids: List[str] = field(default_factory=list)
    match_count: int = 0
    description: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "tournament_ids": self.tournament_ids,
            "match_count": self.match_count,
            "description": self.description,
            "created_at": self.created_at,
        }
