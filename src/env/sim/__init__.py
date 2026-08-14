"""CR-Pipeline: Simulation Environment.

Provides the low-fidelity Clash Royale simulation engine for training
evolutionary agents. Includes entity definitions, action space, game state,
the core simulation engine, and parallel evaluation infrastructure.

Contents:
- Unit status effects (stun, slow, poison) that actually modify the unit
- King tower activation (on damage, or on losing a princess tower)
- Air/ground targeting driven by card definitions
- Crown-based match scoring with overtime
- Card registry of 140 entries at Level 11 stats
- Death spawns (Golem -> Golem Minis) declared per card
- Shaped rewards (crowns, tower lead, board health, territory, elixir)
- Multiple scripted opponents plus genome-driven self-play
- Replay data with full state snapshots
"""

from .engine import (
    SimulationEngine, 
    SimulationStepResult, 
    UnitState, 
    UnitStatus,
    OpponentStrategy,
)
from .entities import (
    CARD_DEFS, TOWER_DEFS, DeploymentZone, EntityType,
    TargetingMode, TowerDefinition, UnitInstance,
    UnitType, get_default_card_registry, get_default_tower_registry,
)
from .actions import Action, ActionSpace, ActionValidator, ActionType
from .state import (
    GameStateSnapshot, UnitState, preprocess_state,
    compute_state_from_arena, compute_enhanced_state,
)
from .parallel_runner import (
    ParallelRunner, MatchResult, WorkerConfig,
)

__all__ = [
    # Engine
    "SimulationEngine",
    "SimulationStepResult",
    "UnitState",
    "UnitStatus",
    "OpponentStrategy",
    # Entities
    "CARD_DEFS",
    "TOWER_DEFS",
    "DeploymentZone",
    "EntityType",
    "TargetingMode",
    "TowerDefinition",
    "UnitInstance",
    "UnitType",
    "get_default_card_registry",
    "get_default_tower_registry",
    # Actions
    "Action",
    "ActionType",
    "ActionSpace",
    "ActionValidator",
    # State
    "GameStateSnapshot",
    "preprocess_state",
    "compute_state_from_arena",
    "compute_enhanced_state",
    # Parallel runner
    "ParallelRunner",
    "MatchResult",
    "WorkerConfig",
]