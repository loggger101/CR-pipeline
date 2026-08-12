"""Hyperparameter configuration management.

Provides:
- Default hyperparameter sets for different training phases
- YAML config loading and validation
- Config merging and override support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ── Default Configurations ───────────────────────────────────────────────────

DEFAULT_EVOLUTION_CONFIG = {
    "population": {
        "size": 200,
        "elite_count": 10,
        "elite_preservation": True,
    },
    "selection": {
        "strategy": "tournament",
        "tournament_size": 5,
        "rank_weight": 1.5,
    },
    "crossover": {
        "strategy": "blend",
        "rate": 0.7,
        "blend_alpha": 0.5,
    },
    "mutation": {
        "strategy": "gaussian",
        "rate": 0.05,
        "std": 0.1,
        "min_std": 0.01,
        "max_std": 0.5,
        "adaptive": False,
    },
    "fitness": {
        "matches_per_agent": 5,
        "match_duration": "full",
        "scoring": {
            "trophy_gain_weight": 0.4,
            "towers_destroyed_weight": 0.3,
            "win_bonus": 0.2,
            "efficiency_weight": 0.1,
        },
        "baseline_opponents": ["random", "greedy", "elite_avg"],
    },
    "checkpoint": {
        "interval": 10,
        "max_checkpoints": 50,
        "format": "pt",
        "include": [
            "population_weights",
            "fitness_history",
            "generation_metadata",
            "best_agent_snapshot",
        ],
    },
    "training": {
        "max_generations": 500,
        "early_stopping": {
            "enabled": True,
            "patience": 30,
            "min_improvement": 0.5,
        },
        "parallel": {
            "workers": 8,
            "batch_size": 50,
            "timeout": 300,
        },
        "logging": {
            "log_interval": 1,
            "save_full_history": True,
        },
    },
}

DEFAULT_SIM_CONFIG = {
    "arena": {
        "width": 8,
        "height": 6,
        "left_lane": [0, 1, 2, 3],
        "right_lane": [4, 5, 6, 7],
        "opponent_territory": [0, 1, 2],
        "player_territory": [3, 4, 5],
        "bridges": [3, 4],
    },
    "game_rules": {
        "match_duration_ticks": 1800,
        "overtime_duration_ticks": 120,
        "overtime_threshold_trophies": 0,
        "elixir_max": 10,
        "elixir_regen_rate": 0.3,
        "elixir_double_overtime": True,
        "deployment_cooldown": 0,
        "win_condition": "trophies",
        "trophy_win": 2,
        "trophy_loss": 2,
        "overtime_trophy_win": 1,
        "overtime_trophy_loss": 1,
    },
    "state_input": {
        "resolution": 64,
        "total_channels": 12,
    },
}

DEFAULT_LIVE_CONFIG = {
    "screen_capture": {
        "target_window": "Clash Royale",
        "capture_method": "mss",
        "resolution": [1280, 720],
        "target_resolution": [256, 256],
        "frame_rate": 15,
        "color_format": "rgb",
    },
    "game_state_extraction": {
        "method": "template_matching",
    },
    "action_mapping": {
        "method": "mouse_click",
        "grid_resolution": [8, 6],
        "deployment_zone": "player_half",
        "debounce_ms": 100,
        "cooldown_validation": True,
    },
    "overlay": {
        "enabled": True,
        "draw_card_selection": True,
        "draw_action_preview": True,
        "fps_display": True,
        "fitness_display": True,
    },
    "performance": {
        "max_latency_ms": 66,
        "buffer_frames": 3,
        "auto_reconnect": True,
    },
}


# ── Phase Configurations ─────────────────────────────────────────────────────

PHASE_CONFIGS = {
    "phase1_baseline": {
        "description": "Random policy baseline in simulation",
        "evolution": {
            "population_size": 50,
            "elite_count": 5,
            "mutation_rate": 0.1,
            "mutation_std": 0.2,
            "max_generations": 10,
            "matches_per_agent": 3,
            "num_workers": 4,
        },
        "sim": {
            "match_duration": "short",
        },
    },
    "phase2_basic": {
        "description": "Evolution in simplified arena",
        "evolution": {
            "population_size": 100,
            "elite_count": 10,
            "mutation_rate": 0.05,
            "mutation_std": 0.1,
            "max_generations": 50,
            "matches_per_agent": 5,
            "num_workers": 8,
        },
        "sim": {
            "match_duration": "full",
        },
    },
    "phase3_advanced": {
        "description": "Evolution in full simulation",
        "evolution": {
            "population_size": 200,
            "elite_count": 20,
            "mutation_rate": 0.03,
            "mutation_std": 0.08,
            "max_generations": 100,
            "matches_per_agent": 10,
            "num_workers": 16,
            "adaptive_mutation": True,
        },
        "sim": {
            "match_duration": "full",
        },
    },
    "phase4_finetune": {
        "description": "Fine-tune on live game",
        "evolution": {
            "population_size": 100,
            "elite_count": 15,
            "mutation_rate": 0.02,
            "mutation_std": 0.05,
            "max_generations": 100,
            "matches_per_agent": 15,
            "num_workers": 8,
            "adaptive_mutation": True,
        },
        "sim": {
            "match_duration": "full",
        },
    },
    "phase5_competitive": {
        "description": "Self-play competitive evolution",
        "evolution": {
            "population_size": 300,
            "elite_count": 30,
            "mutation_rate": 0.02,
            "mutation_std": 0.05,
            "max_generations": 200,
            "matches_per_agent": 20,
            "num_workers": 32,
            "adaptive_mutation": True,
            "crossover_rate": 0.8,
        },
        "sim": {
            "match_duration": "full",
        },
    },
}


# ── Config Loader ────────────────────────────────────────────────────────────


class HyperparamsLoader:
    """Loads and manages hyperparameter configurations.

    Supports:
    - Loading from YAML files
    - Merging multiple configs
    - Override values
    - Phase-specific presets
    - Validation
    """

    def __init__(self):
        self.configs: Dict[str, Any] = {}

    def load_evolution_config(self, path: Optional[str] = None) -> Dict:
        """Load evolution hyperparameters.

        Args:
            path: Optional YAML file path. Uses defaults if not provided.

        Returns:
            Merged evolution configuration.
        """
        if path:
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            config = self._deep_merge(DEFAULT_EVOLUTION_CONFIG, user_config)
        else:
            config = DEFAULT_EVOLUTION_CONFIG.copy()

        self.configs["evolution"] = config
        return config

    def load_sim_config(self, path: Optional[str] = None) -> Dict:
        """Load simulation configuration.

        Args:
            path: Optional YAML file path.

        Returns:
            Merged simulation configuration.
        """
        if path:
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            config = self._deep_merge(DEFAULT_SIM_CONFIG, user_config)
        else:
            config = DEFAULT_SIM_CONFIG.copy()

        self.configs["sim"] = config
        return config

    def load_live_config(self, path: Optional[str] = None) -> Dict:
        """Load live game configuration.

        Args:
            path: Optional YAML file path.

        Returns:
            Merged live game configuration.
        """
        if path:
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            config = self._deep_merge(DEFAULT_LIVE_CONFIG, user_config)
        else:
            config = DEFAULT_LIVE_CONFIG.copy()

        self.configs["live"] = config
        return config

    def load_phase_config(self, phase: str) -> Dict:
        """Load a phase-specific configuration preset.

        Args:
            phase: Phase identifier (e.g., "phase1_baseline").

        Returns:
            Phase configuration dictionary.
        """
        if phase not in PHASE_CONFIGS:
            raise ValueError(f"Unknown phase: {phase}. "
                           f"Available: {list(PHASE_CONFIGS.keys())}")

        config = PHASE_CONFIGS[phase].copy()
        self.configs["phase"] = config
        return config

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply override values to loaded configs.

        Args:
            overrides: Dictionary of override values.
        """
        for key, value in overrides.items():
            parts = key.split(".")
            target = self.configs
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all loaded configurations.

        Returns:
            Dictionary of all configurations.
        """
        return self.configs

    def validate_config(self, config: Dict, config_type: str = "evolution") -> bool:
        """Validate a configuration dictionary.

        Args:
            config: Configuration to validate.
            config_type: Type of config ("evolution", "sim", "live").

        Returns:
            True if valid.

        Raises:
            ValueError: If validation fails.
        """
        if config_type == "evolution":
            self._validate_evolution_config(config)
        elif config_type == "sim":
            self._validate_sim_config(config)
        elif config_type == "live":
            self._validate_live_config(config)

        return True

    def _validate_evolution_config(self, config: Dict) -> None:
        """Validate evolution configuration."""
        pop = config.get("population", {})
        if pop.get("size", 0) <= 0:
            raise ValueError("population.size must be positive")
        if pop.get("elite_count", 0) >= pop.get("size", 0):
            raise ValueError("elite_count must be less than population.size")

        sel = config.get("selection", {})
        if sel.get("strategy") not in ("tournament", "rank", "roulette"):
            raise ValueError("selection.strategy must be tournament, rank, or roulette")

        cr = config.get("crossover", {})
        if cr.get("strategy") not in ("blend", "single_point", "uniform"):
            raise ValueError("crossover.strategy must be blend, single_point, or uniform")

        mut = config.get("mutation", {})
        if mut.get("strategy") not in ("gaussian", "uniform", "adaptive"):
            raise ValueError("mutation.strategy must be gaussian, uniform, or adaptive")

        train = config.get("training", {})
        if train.get("max_generations", 0) <= 0:
            raise ValueError("training.max_generations must be positive")

    def _validate_sim_config(self, config: Dict) -> None:
        """Validate simulation configuration."""
        arena = config.get("arena", {})
        if arena.get("width", 0) <= 0 or arena.get("height", 0) <= 0:
            raise ValueError("arena dimensions must be positive")

        rules = config.get("game_rules", {})
        if rules.get("match_duration_ticks", 0) <= 0:
            raise ValueError("match_duration_ticks must be positive")
        if rules.get("elixir_max", 0) <= 0:
            raise ValueError("elixir_max must be positive")

    def _validate_live_config(self, config: Dict) -> None:
        """Validate live game configuration."""
        capture = config.get("screen_capture", {})
        if capture.get("capture_method") not in ("mss", "pyautogui", "dxcam"):
            raise ValueError("capture_method must be mss, pyautogui, or dxcam")

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary.
            override: Override dictionary (takes precedence).

        Returns:
            Merged dictionary.
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
