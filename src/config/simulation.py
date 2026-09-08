"""Simulation configuration loader.

Reads ``configs/sim_game.yaml`` and validates it, providing a dataclass that
can be passed directly to :class:`src.env.sim.SimulationEngine`.  This turns the
previously documentation-only YAML file into an active config source so that
tower stats, elixir rates, match duration, etc. can be tuned without touching
engine code.

Usage::

    from src.config import SimulationConfig

    cfg = SimulationConfig.from_file("configs/sim_game.yaml")
    engine = SimulationEngine(
        deck=["knight", "archers"],
        opponent_deck=["hog_rider", "skeletons"],
        **cfg.to_engine_kwargs(),
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ── Default sim_game.yaml contents (mirrors engine defaults) ────────────────

_DEFAULT_SIM_CONFIG: Dict[str, Any] = {
    "arena": {
        "width": 8,
        "height": 6,
        "left_lane": [0, 1, 2, 3],
        "right_lane": [4, 5, 6, 7],
        "opponent_territory": [0, 1, 2],
        "player_territory": [3, 4, 5],
        "bridges": [3, 4],
    },
    "towers": {
        "opponent_princess_left": {
            "col": 2.0, "row": 1.0,
            "hp": 2534, "damage": 109, "attack_speed": 0.8, "range": 7.5,
            "crown_reward": 1,
        },
        "opponent_princess_right": {
            "col": 5.0, "row": 1.0,
            "hp": 2534, "damage": 109, "attack_speed": 0.8, "range": 7.5,
            "crown_reward": 1,
        },
        "opponent_king": {
            "col": 3.5, "row": 0.0,
            "hp": 4008, "damage": 109, "attack_speed": 1.0, "range": 7.0,
            "crown_reward": 3, "starts_inactive": True,
        },
        "player_princess_left": {
            "col": 2.0, "row": 4.0,
            "hp": 2534, "damage": 109, "attack_speed": 0.8, "range": 7.5,
            "crown_reward": 1,
        },
        "player_princess_right": {
            "col": 5.0, "row": 4.0,
            "hp": 2534, "damage": 109, "attack_speed": 0.8, "range": 7.5,
            "crown_reward": 1,
        },
        "player_king": {
            "col": 3.5, "row": 5.0,
            "hp": 4008, "damage": 109, "attack_speed": 1.0, "range": 7.0,
            "crown_reward": 3, "starts_inactive": True,
        },
    },
    "game_rules": {
        "match_duration_ticks": 1800,   # 180 s at 10 ticks/s (real regulation)
        "overtime_ticks": 600,          # up to 60 s sudden-death OT (real rule)
        "double_elixir_overtime": True,
        "elixir_regen_rate": None,      # None → default (1/2.8 per tick)
        "elixir_max": 10,
    },
}


def _resolve_path(path: str) -> Path:
    """Resolve a sim_game.yaml path, falling back to the project default."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    # Try relative to common locations
    for candidate in (
        Path(__file__).resolve().parent.parent / ".." / "configs" / "sim_game.yaml",
        Path("configs" / "sim_game.yaml").resolve(),
    ):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"sim_game.yaml not found at {path}; tried: {[str(c) for c in [p, *candidate.parents]]}"
    )


@dataclass(frozen=True)
class SimulationConfig:
    """Validated simulation configuration loaded from sim_game.yaml.

    All fields have sensible defaults that mirror the engine's built-in values,
    so an empty YAML file still produces a valid config.

    Attributes:
        arena_width: Grid columns (default 8).
        arena_height: Grid rows (default 6).
        bridges: Column indices where troops cross the river.
        match_duration_ticks: Regulation length in ticks (default 1800 = 180 s).
        overtime_ticks: Max extra ticks for sudden-death OT (default 600 = 60 s).
        double_elixir_overtime: Whether elixir doubles during overtime.
        elixir_regen_rate: Elixir per tick; ``None`` → default (1/2.8).
        elixir_max: Maximum elixir capacity.
    """
    arena_width: int = 8
    arena_height: int = 6
    bridges: List[int] = field(default_factory=lambda: [3, 4])
    match_duration_ticks: int = 1800
    overtime_ticks: int = 600
    double_elixir_overtime: bool = True
    elixir_regen_rate: Optional[float] = None
    elixir_max: int = 10

    @classmethod
    def from_file(cls, path: str = "configs/sim_game.yaml") -> "SimulationConfig":
        """Load and validate a ``sim_game.yaml`` file.

        Args:
            path: Path to the YAML config. Falls back to project default.

        Returns:
            Validated SimulationConfig.

        Raises:
            FileNotFoundError: if no valid sim_game.yaml is found.
            yaml.YAMLError: on malformed YAML.
            ValueError: on invalid values (e.g. negative HP).
        """
        resolved = _resolve_path(path)
        logger.info(f"Loading simulation config from {resolved}")

        with open(resolved) as f:
            raw = yaml.safe_load(f) or {}

        # Merge with defaults so partial files still produce valid configs
        merged = _deep_merge(_DEFAULT_SIM_CONFIG, raw)
        return cls._from_dict(merged)

    @classmethod
    def from_defaults(cls) -> "SimulationConfig":
        """Return a config built entirely from the engine's default values."""
        return cls._from_dict(_DEFAULT_SIM_CONFIG)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        rules = data.get("game_rules", {})
        arena = data.get("arena", {})

        regen = rules.get("elixir_regen_rate")
        if regen is not None and regen <= 0:
            raise ValueError(f"elixir_regen_rate must be positive, got {regen}")

        duration = rules.get("match_duration_ticks", 1800)
        if duration <= 0:
            raise ValueError(f"match_duration_ticks must be positive, got {duration}")

        return cls(
            arena_width=arena.get("width", 8),
            arena_height=arena.get("height", 6),
            bridges=rules.get("bridges", [3, 4]),
            match_duration_ticks=int(duration),
            # Older config files used the longer key name; accept both.
            overtime_ticks=int(
                rules.get("overtime_ticks",
                          rules.get("overtime_duration_ticks", 600))
            ),
            double_elixir_overtime=bool(rules.get("double_elixir_overtime", True)),
            elixir_regen_rate=regen,
            elixir_max=int(rules.get("elixir_max", 10)),
        )

    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Convert to keyword arguments for ``SimulationEngine.__init__``."""
        kwargs = {
            "match_duration_ticks": self.match_duration_ticks,
            "overtime_ticks": self.overtime_ticks,
            "double_elixir_overtime": self.double_elixir_overtime,
            "elixir_max": self.elixir_max,
        }
        if self.elixir_regen_rate is not None:
            kwargs["elixir_regen_rate"] = self.elixir_regen_rate
        return kwargs

    def summary(self) -> str:
        """Human-readable summary of the config."""
        regen_str = (f"{self.elixir_regen_rate:.4f}/tick"
                     if self.elixir_regen_rate is not None else "default")
        return (
            f"SimulationConfig(duration={self.match_duration_ticks} ticks, "
            f"overtime={self.overtime_ticks}, elixir={regen_str}, "
            f"max={self.elixir_max})"
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Lists are replaced entirely (not concatenated); dicts are merged recursively.
    """
    result = dict(base)
    for key, val in override.items():
        if (key in result and isinstance(result[key], dict)
                and isinstance(val, dict)):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
