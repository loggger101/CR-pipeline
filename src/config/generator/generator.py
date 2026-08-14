"""Configuration generator for CR-Pipeline.

Generates validated YAML configurations for:
- Evolutionary training
- Simulation engine
- Tournament evaluation
- Live-game interaction
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigPreset(Enum):
    """Built-in configuration presets."""
    EVOLUTION_FAST = auto()
    EVOLUTION_STANDARD = auto()
    EVOLUTION_COMPETITIVE = auto()
    SIMULATION_QUICK = auto()
    SIMULATION_STANDARD = auto()
    SIMULATION_DETAILED = auto()
    TOURNAMENT_ROUND_ROBIN = auto()
    TOURNAMENT_ELIMINATION = auto()
    TOURNAMENT_LEAGUE = auto()


@dataclass
class ConfigGenerator:
    """Generates and validates CR-Pipeline configurations."""

    output_dir: str = "configs/generated"

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, preset: ConfigPreset, name: str = None,
                 overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a configuration from a preset.

        Args:
            preset: Configuration preset.
            name: Output filename (without .yaml).
            overrides: Key-value overrides.

        Returns:
            Generated configuration dictionary.
        """
        generators = {
            ConfigPreset.EVOLUTION_FAST: self._gen_evolution_fast,
            ConfigPreset.EVOLUTION_STANDARD: self._gen_evolution_standard,
            ConfigPreset.EVOLUTION_COMPETITIVE: self._gen_evolution_competitive,
            ConfigPreset.SIMULATION_QUICK: self._gen_sim_quick,
            ConfigPreset.SIMULATION_STANDARD: self._gen_sim_standard,
            ConfigPreset.SIMULATION_DETAILED: self._gen_sim_detailed,
            ConfigPreset.TOURNAMENT_ROUND_ROBIN: self._gen_tournament_rr,
            ConfigPreset.TOURNAMENT_ELIMINATION: self._gen_tournament_elim,
            ConfigPreset.TOURNAMENT_LEAGUE: self._gen_tournament_league,
        }

        generator = generators.get(preset)
        if not generator:
            raise ValueError(f"Unknown preset: {preset}")

        config = generator()
        if overrides:
            config = self._merge_overrides(config, overrides)

        if name is None:
            name = preset.name.lower()

        # Save to file
        output_path = Path(self.output_dir) / f"{name}.yaml"
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Generated config: {output_path}")
        return config

    def _gen_evolution_fast(self) -> Dict[str, Any]:
        return {
            "population": {"size": 50, "elite_count": 5, "elite_preservation": True},
            "selection": {"strategy": "tournament", "tournament_size": 3, "rank_weight": 1.5},
            "crossover": {"strategy": "blend", "rate": 0.7, "blend_alpha": 0.5},
            "mutation": {"strategy": "gaussian", "rate": 0.05, "std": 0.1, "adaptive": False},
            "fitness": {"matches_per_agent": 3, "match_duration": "short", "scoring": {
                "trophy_gain_weight": 0.4, "towers_destroyed_weight": 0.3,
                "win_bonus": 0.2, "efficiency_weight": 0.1,
            }, "baseline_opponents": ["random", "greedy"]},
            "training": {"max_generations": 50, "early_stopping": {"enabled": True, "patience": 10, "min_improvement": 0.5},
                "parallel": {"workers": 4, "batch_size": 25, "timeout": 300}, "logging": {"log_interval": 1, "save_full_history": True}},
            "tournament": {"enabled": True, "format": "round_robin", "matches_per_pair": 2, "elite_fraction": 0.1},
            "curriculum": {"enabled": True, "convergence_threshold": 0.01},
            "monitoring": {"enabled": True, "sample_interval": 1.0},
            "alerting": {"enabled": True, "log_path": "runs/alerts.log"},
            "registry": {"enabled": True, "dir": "runs/model_registry"},
            "seed": 42, "runs_dir": "runs",
        }

    def _gen_evolution_standard(self) -> Dict[str, Any]:
        return {
            "population": {"size": 200, "elite_count": 10, "elite_preservation": True},
            "selection": {"strategy": "tournament", "tournament_size": 5, "rank_weight": 1.5},
            "crossover": {"strategy": "blend", "rate": 0.7, "blend_alpha": 0.5},
            "mutation": {"strategy": "gaussian", "rate": 0.05, "std": 0.1, "min_std": 0.01, "max_std": 0.5, "adaptive": False},
            "fitness": {"matches_per_agent": 5, "match_duration": "full", "scoring": {
                "trophy_gain_weight": 0.4, "towers_destroyed_weight": 0.3,
                "win_bonus": 0.2, "efficiency_weight": 0.1,
            }, "baseline_opponents": ["random", "greedy", "elite_avg"]},
            "checkpoint": {"interval": 10, "max_checkpoints": 50, "format": "pt"},
            "training": {"max_generations": 200, "early_stopping": {"enabled": True, "patience": 30, "min_improvement": 0.5},
                "parallel": {"workers": 8, "batch_size": 50, "timeout": 300}, "logging": {"log_interval": 1, "save_full_history": True, "monitor_resources": True}},
            "tournament": {"enabled": True, "format": "round_robin", "matches_per_pair": 4, "elite_fraction": 0.1, "elo_k_factor": 32},
            "curriculum": {"enabled": True, "convergence_threshold": 0.01},
            "diversity": {"preservation": True, "threshold": 0.5, "novelty_search": False},
            "monitoring": {"enabled": True, "sample_interval": 1.0},
            "alerting": {"enabled": True, "log_path": "runs/alerts.log"},
            "registry": {"enabled": True, "dir": "runs/model_registry"},
            "seed": 42, "runs_dir": "runs",
        }

    def _gen_evolution_competitive(self) -> Dict[str, Any]:
        return {
            "population": {"size": 300, "elite_count": 15, "elite_preservation": True},
            "selection": {"strategy": "tournament_elite", "tournament_size": 7, "rank_weight": 2.0},
            "crossover": {"strategy": "blend", "rate": 0.8, "blend_alpha": 0.3},
            "mutation": {"strategy": "adaptive", "rate": 0.03, "std": 0.15, "min_std": 0.01, "max_std": 0.5, "adaptive": True},
            "fitness": {"matches_per_agent": 10, "match_duration": "full", "scoring": {
                "trophy_gain_weight": 0.4, "towers_destroyed_weight": 0.3,
                "win_bonus": 0.2, "efficiency_weight": 0.1,
            }, "baseline_opponents": ["elite_avg", "greedy", "aggressive"]},
            "checkpoint": {"interval": 5, "max_checkpoints": 100, "format": "pt"},
            "training": {"max_generations": 500, "early_stopping": {"enabled": True, "patience": 50, "min_improvement": 0.5},
                "parallel": {"workers": 16, "batch_size": 50, "timeout": 600}, "logging": {"log_interval": 1, "save_full_history": True, "monitor_resources": True}},
            "tournament": {"enabled": True, "format": "double_elim", "matches_per_pair": 6, "elite_fraction": 0.15, "elo_k_factor": 32},
            "curriculum": {"enabled": True, "convergence_threshold": 0.005},
            "diversity": {"preservation": True, "threshold": 0.5, "novelty_search": True, "novelty_window": 50},
            "adaptive_population": {"enabled": True, "min_size": 100, "max_size": 400, "growth_rate": 0.5},
            "monitoring": {"enabled": True, "sample_interval": 0.5},
            "alerting": {"enabled": True, "log_path": "runs/alerts.log"},
            "registry": {"enabled": True, "dir": "runs/model_registry"},
            "seed": 42, "runs_dir": "runs",
        }

    def _gen_sim_quick(self) -> Dict[str, Any]:
        return {
            "arena": {"width": 8, "height": 6, "bridges": [3, 4]},
            "game_rules": {"match_duration_ticks": 600, "elixir_max": 10, "elixir_regen_rate": 0.3, "win_condition": "trophies"},
            "augmentation": {"enabled": True, "intensity": 0.3},
        }

    def _gen_sim_standard(self) -> Dict[str, Any]:
        return {
            "arena": {"width": 8, "height": 6, "left_lane": [0, 1, 2, 3], "right_lane": [4, 5, 6, 7], "bridges": [3, 4]},
            "game_rules": {"match_duration_ticks": 1800, "overtime_duration_ticks": 120, "elixir_max": 10, "elixir_regen_rate": 0.3, "elixir_double_overtime": True, "win_condition": "trophies"},
            "state_input": {"resolution": 64, "total_channels": 11},
            "augmentation": {"enabled": True, "intensity": 0.5, "strategies": {"deck_composition": True, "card_order": True, "opponent_strategy": True}},
        }

    def _gen_sim_detailed(self) -> Dict[str, Any]:
        return {
            "arena": {"width": 8, "height": 6, "left_lane": [0, 1, 2, 3], "right_lane": [4, 5, 6, 7], "bridges": [3, 4]},
            "game_rules": {"match_duration_ticks": 1800, "overtime_duration_ticks": 120, "elixir_max": 10, "elixir_regen_rate": 0.3, "elixir_double_overtime": True, "win_condition": "trophies"},
            "state_input": {"resolution": 64, "total_channels": 11},
            "augmentation": {"enabled": True, "intensity": 0.8, "strategies": {"deck_composition": True, "card_order": True, "opponent_strategy": True, "game_conditions": True, "elixir_advantage": True, "timing": True}},
        }

    def _gen_tournament_rr(self) -> Dict[str, Any]:
        return {
            "format": "round_robin", "matches_per_pair": 4, "elite_fraction": 0.1, "elo_k_factor": 32,
            "collection": {"enabled": True, "save_bracket": True, "save_elo": True, "save_h2h": True},
        }

    def _gen_tournament_elim(self) -> Dict[str, Any]:
        return {
            "format": "single_elimination", "matches_per_pair": 3, "elite_fraction": 0.15, "elo_k_factor": 32,
            "collection": {"enabled": True, "save_bracket": True, "save_elo": True, "save_h2h": True},
        }

    def _gen_tournament_league(self) -> Dict[str, Any]:
        return {
            "format": "league", "matches_per_pair": 6, "elite_fraction": 0.2, "elo_k_factor": 32,
            "collection": {"enabled": True, "save_bracket": True, "save_elo": True, "save_h2h": True},
        }

    def _merge_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge overrides into config."""
        result = dict(config)
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                d = result
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            else:
                result[key] = value
        return result


def generate_evolution_config(preset: str = "standard", name: str = None,
                              overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate an evolution config.

    Args:
        preset: Preset name (fast, standard, competitive).
        name: Output filename.
        overrides: Key-value overrides.

    Returns:
        Generated configuration.
    """
    preset_map = {
        "fast": ConfigPreset.EVOLUTION_FAST,
        "standard": ConfigPreset.EVOLUTION_STANDARD,
        "competitive": ConfigPreset.EVOLUTION_COMPETITIVE,
    }
    generator = ConfigGenerator()
    return generator.generate(preset_map[preset], name, overrides)


def generate_simulation_config(preset: str = "standard", name: str = None,
                               overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate a simulation config."""
    preset_map = {
        "quick": ConfigPreset.SIMULATION_QUICK,
        "standard": ConfigPreset.SIMULATION_STANDARD,
        "detailed": ConfigPreset.SIMULATION_DETAILED,
    }
    generator = ConfigGenerator()
    return generator.generate(preset_map[preset], name, overrides)


def generate_tournament_config(preset: str = "round_robin", name: str = None,
                               overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate a tournament config."""
    preset_map = {
        "round_robin": ConfigPreset.TOURNAMENT_ROUND_ROBIN,
        "elimination": ConfigPreset.TOURNAMENT_ELIMINATION,
        "league": ConfigPreset.TOURNAMENT_LEAGUE,
    }
    generator = ConfigGenerator()
    return generator.generate(preset_map[preset], name, overrides)
