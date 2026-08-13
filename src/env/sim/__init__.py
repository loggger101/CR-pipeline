"""CR-Pipeline: Simulation Environment.

Provides the low-fidelity Clash Royale simulation engine for training
evolutionary agents. Includes entity definitions, action space, game state,
the core simulation engine, and parallel evaluation infrastructure.
"""

from .engine import SimulationEngine, SimulationStepResult
from .entities import (
    CARD_DEFS, TOWER_DEFS, DeployZone, EntityType,
    TargetPreference, TowerDefinition, UnitDefinition,
    get_card_def, get_tower_def,
)
from .actions import Action, ActionSpace, ActionValidator, ActionType
from .state import (
    GameStateSnapshot, UnitState, preprocess_state,
    compute_state_from_arena,
)
from .parallel_runner import (
    ParallelRunner, MatchResult, WorkerConfig,
)

__all__ = [
    # Engine
    "SimulationEngine",
    "SimulationStepResult",
    # Entities
    "CARD_DEFS",
    "TOWER_DEFS",
    "DeployZone",
    "EntityType",
    "TargetPreference",
    "TowerDefinition",
    "UnitDefinition",
    "get_card_def",
    "get_tower_def",
    # Actions
    "Action",
    "ActionSpace",
    "ActionValidator",
    "ActionType",
    # State
    "GameStateSnapshot",
    "UnitState",
    "preprocess_state",
    "compute_state_from_arena",
    # Parallel
    "ParallelRunner",
    "MatchResult",
    "WorkerConfig",
]
