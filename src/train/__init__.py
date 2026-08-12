"""CR-Pipeline: Training Pipeline.

Provides:
- EvolutionTrainer: Main training loop orchestrator
- FitnessEvaluator: Agent evaluation against opponents
- CheckpointManager: Save/load training state
- HyperparamsLoader: Configuration management with phase presets
"""

from .trainer import EvolutionTrainer, TrainingConfig
from .evaluator import FitnessEvaluator, TournamentResult
from .checkpoint import CheckpointManager
from .hyperparams import (
    HyperparamsLoader,
    DEFAULT_EVOLUTION_CONFIG,
    DEFAULT_SIM_CONFIG,
    DEFAULT_LIVE_CONFIG,
    PHASE_CONFIGS,
)

__all__ = [
    "EvolutionTrainer",
    "TrainingConfig",
    "FitnessEvaluator",
    "TournamentResult",
    "CheckpointManager",
    "HyperparamsLoader",
    "DEFAULT_EVOLUTION_CONFIG",
    "DEFAULT_SIM_CONFIG",
    "DEFAULT_LIVE_CONFIG",
    "PHASE_CONFIGS",
]
