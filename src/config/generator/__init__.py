"""Configuration generator for CR-Pipeline.

Provides:
- Interactive config generation
- Preset templates
- Config validation
- Config merging
- Config diffing
"""

from .generator import (
    ConfigGenerator,
    ConfigPreset,
    generate_evolution_config,
    generate_simulation_config,
    generate_tournament_config,
)

__all__ = [
    "ConfigGenerator",
    "ConfigPreset",
    "generate_evolution_config",
    "generate_simulation_config",
    "generate_tournament_config",
]
