"""Entity definitions for the Clash Royale simulation engine.

Defines cards, towers, units, and their properties. Each card type maps to
a UnitDefinition (stats) and deployment constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple


class EntityType(Enum):
    """Categories of game entities."""
    TOWER = auto()
    UNIT = auto()
    SPELL = auto()


class TargetPreference(Enum):
    """What a unit/tower prioritizes attacking."""
    GROUND = auto()
    AIR = auto()
    ANY = auto()


class DeployZone(Enum):
    """Where a card can be placed on the grid."""
    GROUND = auto()
    AIR = auto()
    SPELL = auto()
    BUILDING = auto()


@dataclass
class TowerDefinition:
    """Static definition for a tower type.

    Attributes:
        name: Display name (e.g. "princess_left").
        hp: Maximum hit points.
        damage: Damage per attack tick.
        range: Attack range in grid cells.
        target_pref: Preferred target type for attacks.
        is_king: Whether this is a king tower.
        is_opponent: Whether this tower belongs to the opponent.
        activation_range: Distance the king tower must be approached
            to activate it (0 = always active).
    """
    name: str
    hp: int
    damage: int
    range: float
    target_pref: TargetPreference
    is_king: bool = False
    is_opponent: bool = True
    activation_range: float = 4.0  # King towers activate when enemies are close

    def __post_init__(self) -> None:
        if self.is_opponent:
            self.name = f"opp_{self.name}"
        else:
            self.name = f"player_{self.name}"


@dataclass
class UnitDefinition:
    """Static definition for a deployable unit type.

    Attributes:
        name: Display name (e.g. "knight").
        hp: Maximum hit points.
        damage: Damage per attack tick.
        range: Attack range in grid cells.
        speed: Movement speed (cells per tick).
        target_pref: Preferred target type for attacks.
        deploy_zone: Where this unit can be placed.
        is_building: Whether this unit acts as a stationary structure.
        aoe: Whether attacks hit all enemies in range.
        spawn_count: Number of units spawned on death (for swarm cards).
        spawn_type: UnitDefinition of spawned units (for swarm cards).
        elixir_cost: Elixir cost to deploy.
    """
    name: str
    hp: int
    damage: int
    range: float
    speed: float
    target_pref: TargetPreference
    deploy_zone: DeployZone
    is_building: bool = False
    aoe: bool = False
    spawn_count: int = 0
    spawn_type: Optional[str] = None
    elixir_cost: int = 3

    def __post_init__(self) -> None:
        # Ensure floating-point speed
        self.speed = float(self.speed)


# ── Card Registry ────────────────────────────────────────────────────────────

CARD_DEFS: Dict[str, UnitDefinition] = {}

def _register(name: str, **kwargs) -> UnitDefinition:
    """Register a card definition and return it."""
    unit = UnitDefinition(**kwargs)
    CARD_DEFS[name] = unit
    return unit


def get_card_def(name: str) -> UnitDefinition:
    """Look up a card definition by name."""
    if name not in CARD_DEFS:
        raise KeyError(f"Unknown card: {name}")
    return CARD_DEFS[name]


# Register all standard cards
def _init_card_defs() -> None:
    _register("knight",
        name="knight", hp=1400, damage=75, range=0.75, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=3)

    _register("archers",
        name="archers", hp=250, damage=60, range=2.5, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        spawn_count=2, spawn_type="archer",
        elixir_cost=3)

    _register("archer",
        name="archer", hp=250, damage=60, range=2.5, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=1)

    _register("giant",
        name="giant", hp=4000, damage=120, range=0.75, speed=0.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=5)

    _register("minions",
        name="minions", hp=200, damage=60, range=1.0, speed=1.5,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.AIR,
        spawn_count=3, spawn_type="minion",
        elixir_cost=3)

    _register("minion",
        name="minion", hp=200, damage=60, range=1.0, speed=1.5,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.AIR,
        elixir_cost=1)

    _register("wizard",
        name="wizard", hp=600, damage=150, range=2.5, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        aoe=True,
        elixir_cost=5)

    _register("mini_pekka",
        name="mini_pekka", hp=1100, damage=300, range=0.75, speed=1.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=4)

    _register("valkyrie",
        name="valkyrie", hp=1600, damage=135, range=0.75, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        aoe=True,
        elixir_cost=4)

    _register("baby_dragon",
        name="baby_dragon", hp=800, damage=100, range=1.5, speed=1.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.AIR,
        aoe=True,
        elixir_cost=4)

    _register("fireball",
        name="fireball", hp=0, damage=500, range=1.5, speed=0.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.SPELL,
        aoe=True,
        elixir_cost=4)

    _register("zap",
        name="zap", hp=0, damage=150, range=2.0, speed=0.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.SPELL,
        aoe=True,
        elixir_cost=2)

    _register("skeletons",
        name="skeletons", hp=100, damage=80, range=0.75, speed=1.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        spawn_count=3, spawn_type="skeleton",
        elixir_cost=3)

    _register("skeleton",
        name="skeleton", hp=100, damage=80, range=0.75, speed=1.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=1)

    _register("barbarians",
        name="barbarians", hp=350, damage=100, range=0.75, speed=1.2,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        spawn_count=5, spawn_type="barbarian",
        elixir_cost=6)

    _register("barbarian",
        name="barbarian", hp=350, damage=100, range=0.75, speed=1.2,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=1)

    _register("musketeer",
        name="musketeer", hp=600, damage=180, range=3.0, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=4)

    _register("prince",
        name="prince", hp=1600, damage=300, range=0.75, speed=2.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=5)

    _register("bomber",
        name="bomber", hp=350, damage=200, range=2.0, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        aoe=True,
        elixir_cost=3)

    _register("goblin",
        name="goblin", hp=250, damage=100, range=0.75, speed=1.8,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=2)

    _register("ice_golem",
        name="ice_golem", hp=1000, damage=80, range=0.75, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=2)

    _register("elixir_golem",
        name="elixir_golem", hp=1000, damage=80, range=0.75, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=3)

    _register("dragon",
        name="dragon", hp=1200, damage=120, range=1.5, speed=1.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.AIR,
        aoe=True,
        elixir_cost=4)

    _register("electro_dragon",
        name="electro_dragon", hp=1400, damage=120, range=1.5, speed=1.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.AIR,
        aoe=True,
        elixir_cost=5)

    _register("inferno_dragon",
        name="inferno_dragon", hp=1000, damage=50, range=1.5, speed=1.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.AIR,
        elixir_cost=4)

    _register("royal_giant",
        name="royal_giant", hp=3000, damage=150, range=4.0, speed=0.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=7)

    _register("witch",
        name="witch", hp=800, damage=100, range=2.5, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=5)

    _register("heal_spell",
        name="heal_spell", hp=0, damage=0, range=2.0, speed=0.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.SPELL,
        elixir_cost=4)

    _register("tornado",
        name="tornado", hp=0, damage=50, range=2.0, speed=0.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.SPELL,
        elixir_cost=3)

    _register("poison_spell",
        name="poison_spell", hp=0, damage=50, range=2.0, speed=0.0,
        target_pref=TargetPreference.ANY, deploy_zone=DeployZone.SPELL,
        elixir_cost=4)

    _register("wall_breakers",
        name="wall_breakers", hp=300, damage=400, range=0.5, speed=2.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=3)

    _register("balloon",
        name="balloon", hp=1200, damage=200, range=1.5, speed=1.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.AIR,
        elixir_cost=5)

    _register("lumberjack",
        name="lumberjack", hp=1600, damage=200, range=0.75, speed=1.8,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=4)

    _register("electro_wizard",
        name="electro_wizard", hp=600, damage=150, range=2.5, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=4)

    _register("mega_knight",
        name="mega_knight", hp=3500, damage=200, range=0.75, speed=1.2,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=7)

    _register("hog_rider",
        name="hog_rider", hp=1600, damage=250, range=1.0, speed=2.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=4)

    _register("golem",
        name="golem", hp=6000, damage=150, range=0.75, speed=0.4,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=8)

    _register("night_witch",
        name="night_witch", hp=800, damage=100, range=2.5, speed=1.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.AIR,
        elixir_cost=4)

    _register("spear_goblins",
        name="spear_goblins", hp=150, damage=100, range=2.5, speed=1.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        spawn_count=3, spawn_type="spear_goblin",
        elixir_cost=2)

    _register("spear_goblin",
        name="spear_goblin", hp=150, damage=100, range=2.5, speed=1.5,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.GROUND,
        elixir_cost=1)

    _register("x_bow",
        name="x_bow", hp=1800, damage=180, range=4.0, speed=0.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.BUILDING,
        is_building=True,
        elixir_cost=6)

    _register("tombstone",
        name="tombstone", hp=800, damage=0, range=1.0, speed=0.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.BUILDING,
        is_building=True,
        spawn_count=3, spawn_type="skeleton",
        elixir_cost=3)

    _register("tesla",
        name="tesla", hp=1200, damage=150, range=2.5, speed=0.0,
        target_pref=TargetPreference.GROUND, deploy_zone=DeployZone.BUILDING,
        is_building=True,
        elixir_cost=4)


_init_card_defs()


# ── Tower Definitions ────────────────────────────────────────────────────────

TOWER_DEFS: Dict[str, TowerDefinition] = {}

def _register_tower(name: str, **kwargs) -> TowerDefinition:
    """Register a tower definition and return it."""
    tower = TowerDefinition(**kwargs)
    TOWER_DEFS[tower.name] = tower
    return tower


def get_tower_def(name: str) -> TowerDefinition:
    """Look up a tower definition by name."""
    if name not in TOWER_DEFS:
        raise KeyError(f"Unknown tower: {name}")
    return TOWER_DEFS[name]


def _init_tower_defs() -> None:
    _register_tower("princess_left",
        name="princess_left", hp=1400, damage=75, range=2.0,
        target_pref=TargetPreference.GROUND, is_king=False, is_opponent=True)
    _register_tower("princess_right",
        name="princess_right", hp=1400, damage=75, range=2.0,
        target_pref=TargetPreference.GROUND, is_king=False, is_opponent=True)
    _register_tower("king",
        name="king", hp=2400, damage=100, range=3.5,
        target_pref=TargetPreference.GROUND, is_king=True, is_opponent=True,
        activation_range=4.0)
    _register_tower("player_princess_left",
        name="princess_left", hp=1400, damage=75, range=2.0,
        target_pref=TargetPreference.GROUND, is_king=False, is_opponent=False)
    _register_tower("player_princess_right",
        name="princess_right", hp=1400, damage=75, range=2.0,
        target_pref=TargetPreference.GROUND, is_king=False, is_opponent=False)
    _register_tower("player_king",
        name="king", hp=2400, damage=100, range=3.5,
        target_pref=TargetPreference.GROUND, is_king=True, is_opponent=False,
        activation_range=4.0)


_init_tower_defs()
