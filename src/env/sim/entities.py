"""
CR-Pipeline: Simulation Entity Definitions

Defines all game entities: cards, units, towers, spells, and their properties.
These definitions drive the simulation engine's behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class EntityType(Enum):
    """Types of entities that can exist in the simulation."""
    UNIT = auto()
    TOWER = auto()
    SPELL = auto()
    ELIXIR_SPRING = auto()


class UnitType(Enum):
    """Behavioral classification of units."""
    MELEE = auto()
    RANGED = auto()
    AIR = auto()
    BUILDING = auto()
    HERO = auto()
    PRINCESS = auto()


class TargetingMode(Enum):
    """What a unit targets when searching for enemies."""
    NEAREST = auto()
    NEAREST_TOWER = auto()
    PATH_FORWARD = auto()
    SPECIFIC = auto()


class DeploymentZone(Enum):
    """Which side of the arena a card can be deployed on."""
    SELF_SIDE = auto()
    OPPONENT_SIDE = auto()
    EITHER_SIDE = auto()


@dataclass
class CardDefinition:
    """
    Complete definition of a playable card.

    Attributes:
        name: Unique identifier for this card.
        display_name: Human-readable name.
        card_type: Whether this is a unit, spell, or building.
        unit_type: Behavioral type for units (melee, ranged, etc.).
        targeting_mode: Default targeting behavior for units.
        cost: Elixir cost to play this card.
        hitpoints: Base health of the unit/tower.
        damage: Damage per attack cycle.
        attack_speed: Seconds between attacks.
        attack_range: Range of the unit's attack in arena units.
        move_speed: Movement speed in arena units per second.
        deployment_zone: Which side of the arena this can be placed on.
        aoe: Whether this card deals area-of-effect damage.
        aoe_radius: Radius of aoe damage if applicable.
        target_ground: Whether this card targets ground units.
        target_air: Whether this card targets air units.
        spell_damage: Damage dealt by spells (for spell cards).
        spell_radius: Radius of spell effect.
        spell_duration: Duration of spell effects (e.g., slow, buff).
        spawn_count: Number of units spawned (for cards like Elixir Golem).
        spawned_unit: Name of the unit spawned by this card.
        max_stack: Maximum number of copies that can be active.
        elixir regen rate: Elixir generation rate for elixir springs.
    """
    name: str
    display_name: str
    card_type: str = "unit"  # "unit", "spell", "building"
    unit_type: Optional[UnitType] = None
    targeting_mode: Optional[TargetingMode] = None
    cost: float = 0.0
    hitpoints: float = 100.0
    damage: float = 10.0
    attack_speed: float = 1.0  # seconds
    attack_range: float = 1.5  # arena units
    move_speed: float = 1.0  # arena units per second
    deployment_zone: DeploymentZone = DeploymentZone.EITHER_SIDE
    aoe: bool = False
    aoe_radius: float = 0.0
    target_ground: bool = True
    target_air: bool = True
    spell_damage: float = 0.0
    spell_radius: float = 0.0
    spell_duration: float = 0.0
    spawn_count: int = 1
    spawned_unit: Optional[str] = None
    max_stack: int = 1
    elixir_rate: float = 0.0

    # Arena position (set at spawn time)
    x: float = 0.0
    y: float = 0.0

    # Runtime state (managed by engine, not persisted)
    current_hp: float = 0.0
    current_target: Optional[str] = None
    last_attack_time: float = 0.0
    is_alive: bool = True

    def __post_init__(self):
        if self.current_hp == 0:
            self.current_hp = self.hitpoints

    def distance_to(self, x: float, y: float) -> float:
        """Euclidean distance from this entity's position to (x, y)."""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def is_within_range(self, x: float, y: float, range_val: float) -> bool:
        """Check if (x, y) is within attack_range of this entity."""
        return self.distance_to(x, y) <= range_val

    def take_damage(self, damage: float) -> bool:
        """Apply damage, return True if this entity died."""
        self.current_hp = max(0.0, self.current_hp - damage)
        if self.current_hp <= 0:
            self.is_alive = False
        return not self.is_alive

    def to_dict(self) -> dict:
        """Serialize to dictionary for checkpointing."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "card_type": self.card_type,
            "unit_type": self.unit_type.name if self.unit_type else None,
            "targeting_mode": self.targeting_mode.name if self.targeting_mode else None,
            "cost": self.cost,
            "hitpoints": self.hitpoints,
            "damage": self.damage,
            "attack_speed": self.attack_speed,
            "attack_range": self.attack_range,
            "move_speed": self.move_speed,
            "deployment_zone": self.deployment_zone.name,
            "aoe": self.aoe,
            "aoe_radius": self.aoe_radius,
            "target_ground": self.target_ground,
            "target_air": self.target_air,
            "spell_damage": self.spell_damage,
            "spell_radius": self.spell_radius,
            "spell_duration": self.spell_duration,
            "spawn_count": self.spawn_count,
            "spawned_unit": self.spawned_unit,
            "max_stack": self.max_stack,
            "elixir_rate": self.elixir_rate,
            "x": self.x,
            "y": self.y,
            "current_hp": self.current_hp,
            "is_alive": self.is_alive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CardDefinition:
        """Deserialize from dictionary."""
        unit_type = None
        if data.get("unit_type"):
            unit_type = UnitType[data["unit_type"]]
        targeting_mode = None
        if data.get("targeting_mode"):
            targeting_mode = TargetingMode[data["targeting_mode"]]
        deployment_zone = DeploymentZone.EITHER_SIDE
        if data.get("deployment_zone"):
            deployment_zone = DeploymentZone[data["deployment_zone"]]
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            card_type=data.get("card_type", "unit"),
            unit_type=unit_type,
            targeting_mode=targeting_mode,
            cost=data.get("cost", 0.0),
            hitpoints=data.get("hitpoints", 100.0),
            damage=data.get("damage", 10.0),
            attack_speed=data.get("attack_speed", 1.0),
            attack_range=data.get("attack_range", 1.5),
            move_speed=data.get("move_speed", 1.0),
            deployment_zone=deployment_zone,
            aoe=data.get("aoe", False),
            aoe_radius=data.get("aoe_radius", 0.0),
            target_ground=data.get("target_ground", True),
            target_air=data.get("target_air", True),
            spell_damage=data.get("spell_damage", 0.0),
            spell_radius=data.get("spell_radius", 0.0),
            spell_duration=data.get("spell_duration", 0.0),
            spawn_count=data.get("spawn_count", 1),
            spawned_unit=data.get("spawned_unit"),
            max_stack=data.get("max_stack", 1),
            elixir_rate=data.get("elixir_rate", 0.0),
        )


@dataclass
class TowerDefinition(CardDefinition):
    """
    Tower entity definition. Towers are stationary units that attack enemies
    within range and have a crown reward when destroyed.
    """
    card_type: str = "tower"
    unit_type: UnitType = UnitType.BUILDING
    targeting_mode: TargetingMode = TargetingMode.NEAREST
    attack_range: float = 6.0  # Towers have longer range
    crown_reward: int = 1  # How many crowns awarded on destruction
    is_king_tower: bool = False
    is_active: bool = False  # King tower becomes active at 1000 elixir

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "crown_reward": self.crown_reward,
            "is_king_tower": self.is_king_tower,
            "is_active": self.is_active,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict) -> TowerDefinition:
        instance = super().from_dict(data)
        instance.crown_reward = data.get("crown_reward", 1)
        instance.is_king_tower = data.get("is_king_tower", False)
        instance.is_active = data.get("is_active", False)
        return instance


@dataclass
class UnitInstance(CardDefinition):
    """
    A spawned unit instance in the simulation. Extends CardDefinition
    with runtime tracking fields specific to active units.
    """
    card_type: str = "unit"
    owner: int = 0  # 0 = player 1, 1 = player 2
    unit_id: str = ""  # Unique ID for this instance
    path_progress: float = 0.0  # Progress along assigned path (0-1)
    is_stunned: bool = False
    stun_remaining: float = 0.0
    is_silenced: bool = False
    silence_remaining: float = 0.0
    speed_modifier: float = 1.0  # Multiplier for current speed
    damage_modifier: float = 1.0  # Multiplier for current damage
    hp_modifier: float = 1.0  # Multiplier for current HP

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "owner": self.owner,
            "unit_id": self.unit_id,
            "path_progress": self.path_progress,
            "is_stunned": self.is_stunned,
            "stun_remaining": self.stun_remaining,
            "is_silenced": self.is_silenced,
            "silence_remaining": self.silence_remaining,
            "speed_modifier": self.speed_modifier,
            "damage_modifier": self.damage_modifier,
        })
        return base

    @classmethod
    def from_dict(cls, data: dict) -> UnitInstance:
        instance = super().from_dict(data)
        instance.owner = data.get("owner", 0)
        instance.unit_id = data.get("unit_id", "")
        instance.path_progress = data.get("path_progress", 0.0)
        instance.is_stunned = data.get("is_stunned", False)
        instance.stun_remaining = data.get("stun_remaining", 0.0)
        instance.is_silenced = data.get("is_silenced", False)
        instance.silence_remaining = data.get("silence_remaining", 0.0)
        instance.speed_modifier = data.get("speed_modifier", 1.0)
        instance.damage_modifier = data.get("damage_modifier", 1.0)
        return instance


# =============================================================================
# Default Card Registry
# =============================================================================

def get_default_card_registry() -> dict[str, CardDefinition]:
    """
    Return a registry of default Clash Royale cards.
    This is a representative subset; the full game has 90+ cards.
    """
    registry: dict[str, CardDefinition] = {}

    # ----- Infantry (Melee) -----
    registry["knight"] = CardDefinition(
        name="knight",
        display_name="Knight",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST,
        cost=3.0,
        hitpoints=1400,
        damage=100,
        attack_speed=1.5,
        attack_range=0.75,
        move_speed=1.2,
        deployment_zone=DeploymentZone.SELF_SIDE,
    )

    registry["giant"] = CardDefinition(
        name="giant",
        display_name="Giant",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0,
        hitpoints=4000,
        damage=120,
        attack_speed=1.5,
        attack_range=0.75,
        move_speed=0.6,
        deployment_zone=DeploymentZone.SELF_SIDE,
    )

    registry["mini_pec"] = CardDefinition(
        name="mini_pekka",
        display_name="Mini P.E.K.K.A",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST,
        cost=4.0,
        hitpoints=900,
        damage=350,
        attack_speed=1.8,
        attack_range=0.75,
        move_speed=1.4,
        deployment_zone=DeploymentZone.SELF_SIDE,
    )

    # ----- Ranged -----
    registry["archers"] = CardDefinition(
        name="archers",
        display_name="Archers",
        card_type="unit",
        unit_type=UnitType.RANGED,
        targeting_mode=TargetingMode.NEAREST,
        cost=3.0,
        hitpoints=280,
        damage=70,
        attack_speed=1.2,
        attack_range=4.5,
        move_speed=1.0,
        deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2,
    )

    registry["musketeer"] = CardDefinition(
        name="musketeer",
        display_name="Musketeer",
        card_type="unit",
        unit_type=UnitType.RANGED,
        targeting_mode=TargetingMode.NEAREST,
        cost=4.0,
        hitpoints=720,
        damage=160,
        attack_speed=1.2,
        attack_range=5.5,
        move_speed=1.0,
        deployment_zone=DeploymentZone.SELF_SIDE,
    )

    registry["wizard"] = CardDefinition(
        name="wizard",
        display_name="Wizard",
        card_type="unit",
        unit_type=UnitType.RANGED,
        targeting_mode=TargetingMode.NEAREST,
        cost=5.0,
        hitpoints=720,
        damage=140,
        attack_speed=1.5,
        attack_range=4.5,
        move_speed=1.0,
        deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True,
        aoe_radius=1.5,
    )

    # ----- Spell Cards -----
    registry["fireball"] = CardDefinition(
        name="fireball",
        display_name="Fireball",
        card_type="spell",
        cost=4.0,
        spell_damage=600,
        spell_radius=2.5,
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    registry["arrows"] = CardDefinition(
        name="arrows",
        display_name="Arrows",
        card_type="spell",
        cost=3.0,
        spell_damage=190,
        spell_radius=3.0,
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    registry["zap"] = CardDefinition(
        name="zap",
        display_name="Zap",
        card_type="spell",
        cost=2.0,
        spell_damage=140,
        spell_radius=2.0,
        spell_duration=0.5,  # Stun duration
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    registry["poison"] = CardDefinition(
        name="poison",
        display_name="Poison",
        card_type="spell",
        cost=4.0,
        spell_damage=60,
        spell_radius=3.0,
        spell_duration=4.0,  # Damage per tick over duration
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    # ----- Special -----
    registry["elixir_golem"] = CardDefinition(
        name="elixir_golem",
        display_name="Elixir Golem",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=2.0,
        hitpoints=600,
        damage=60,
        attack_speed=1.5,
        attack_range=0.75,
        move_speed=1.0,
        deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2,
        spawned_unit="minion",
    )

    registry["minion"] = CardDefinition(
        name="minion",
        display_name="Minion",
        card_type="unit",
        unit_type=UnitType.AIR,
        targeting_mode=TargetingMode.NEAREST,
        cost=1.5,
        hitpoints=180,
        damage=50,
        attack_speed=1.2,
        attack_range=1.0,
        move_speed=1.5,
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    registry["barbarian"] = CardDefinition(
        name="barbarian",
        display_name="Barbarian",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST,
        cost=3.0,
        hitpoints=380,
        damage=50,
        attack_speed=1.0,
        attack_range=0.75,
        move_speed=1.3,
        deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3,
    )

    registry["valkyrie"] = CardDefinition(
        name="valkyrie",
        display_name="Valkyrie",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST,
        cost=4.0,
        hitpoints=1400,
        damage=160,
        attack_speed=1.5,
        attack_range=1.0,
        move_speed=1.0,
        deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True,
        aoe_radius=1.0,
    )

    registry["skeleton_army"] = CardDefinition(
        name="skeleton_army",
        display_name="Skeleton Army",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST,
        cost=3.0,
        hitpoints=60,
        damage=40,
        attack_speed=1.0,
        attack_range=0.75,
        move_speed=1.3,
        deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=8,
        spawned_unit="skeleton",
    )

    registry["skeleton"] = CardDefinition(
        name="skeleton",
        display_name="Skeleton",
        card_type="unit",
        unit_type=UnitType.MELEE,
        targeting_mode=TargetingMode.NEAREST,
        cost=0.5,
        hitpoints=60,
        damage=40,
        attack_speed=1.0,
        attack_range=0.75,
        move_speed=1.3,
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    return registry


def get_default_tower_registry() -> dict[str, TowerDefinition]:
    """Return default tower definitions for both players."""
    towers: dict[str, TowerDefinition] = {}

    # Princess towers (left, right) - shared definition
    princess_tower = TowerDefinition(
        name="princess_tower",
        display_name="Princess Tower",
        hitpoints=1400,
        damage=70,
        attack_speed=1.2,
        attack_range=6.0,
        crown_reward=1,
        is_king_tower=False,
    )

    king_tower = TowerDefinition(
        name="king_tower",
        display_name="King Tower",
        hitpoints=2400,
        damage=90,
        attack_speed=1.2,
        attack_range=7.0,
        crown_reward=3,
        is_king_tower=True,
        is_active=False,
    )

    # Player 1 towers (bottom side)
    towers["p1_princess_left"] = TowerDefinition(
        **princess_tower.to_dict(),
        name="p1_princess_left",
        x=-3.0,
        y=8.0,
    )
    towers["p1_princess_right"] = TowerDefinition(
        **princess_tower.to_dict(),
        name="p1_princess_right",
        x=3.0,
        y=8.0,
    )
    towers["p1_king"] = TowerDefinition(
        **king_tower.to_dict(),
        name="p1_king",
        x=0.0,
        y=10.0,
    )

    # Player 2 towers (top side)
    towers["p2_princess_left"] = TowerDefinition(
        **princess_tower.to_dict(),
        name="p2_princess_left",
        x=-3.0,
        y=-8.0,
    )
    towers["p2_princess_right"] = TowerDefinition(
        **princess_tower.to_dict(),
        name="p2_princess_right",
        x=3.0,
        y=-8.0,
    )
    towers["p2_king"] = TowerDefinition(
        **king_tower.to_dict(),
        name="p2_king",
        x=0.0,
        y=-10.0,
    )

    return towers
