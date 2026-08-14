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
    SWARM = auto()       # Multiple small units (skeleton army, etc.)
    TANK = auto()        # High HP, targets buildings (giant, etc.)


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
    spell_type: str = "damage"  # damage, stun, poison, heal, clone, freeze, tornado, curse, root, deliver, graveyard, rage, void, pushback
    spawn_count: int = 1
    spawned_unit: Optional[str] = None
    max_stack: int = 1
    elixir_rate: float = 0.0
    # Units produced when this card dies (Golem -> Golem Minis, Lava Hound ->
    # Lava Pups). Kept separate from spawn_count/spawned_unit, which mean
    # "this card deploys as N copies of X" (Minions, Skeleton Army). Folding
    # both into one pair made the engine deploy a Golem as two Golem Minis and
    # a Lava Hound as a single Lava Pup, discarding the tank entirely.
    death_spawn_count: int = 0
    death_spawned_unit: Optional[str] = None

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
            "death_spawn_count": self.death_spawn_count,
            "death_spawned_unit": self.death_spawned_unit,
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
            death_spawn_count=data.get("death_spawn_count", 0),
            death_spawned_unit=data.get("death_spawned_unit"),
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
    Comprehensive registry of all Clash Royale cards at level 11 stats.
    Includes 126+ cards organized by elixir cost (1-9), with all traits and abilities.
    Stats sourced from deckshop.pro and clashroyale.fandom.com (Friendly Level 11).
    """
    registry: dict[str, CardDefinition] = {}

    # =========================================================================
    # 1 ELIXIR CARDS
    # =========================================================================
    registry["skeletons"] = CardDefinition(
        name="skeletons", display_name="Skeletons", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=1.0, hitpoints=81, damage=81, attack_speed=0.37,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="skeleton",
    )
    registry["skeleton"] = CardDefinition(
        name="skeleton", display_name="Skeleton", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=81, damage=81, attack_speed=0.37,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["skull"] = CardDefinition(
        name="skull", display_name="Skull", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=30, damage=30, attack_speed=1.0,
        attack_range=0.5, move_speed=0.0, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["ice_spirit"] = CardDefinition(
        name="ice_spirit", display_name="Ice Spirit", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=1.0, hitpoints=230, damage=110, attack_speed=1.5,
        attack_range=0.75, move_speed=1.8, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["fire_spirit"] = CardDefinition(
        name="fire_spirit", display_name="Fire Spirit", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=1.0, hitpoints=230, damage=207, attack_speed=1.5,
        attack_range=0.75, move_speed=2.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["electro_spirit"] = CardDefinition(
        name="electro_spirit", display_name="Electro Spirit", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=1.0, hitpoints=230, damage=99, attack_speed=1.5,
        attack_range=0.75, move_speed=1.8, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["heal_spirit"] = CardDefinition(
        name="heal_spirit", display_name="Heal Spirit", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=1.0, hitpoints=190, damage=91, attack_speed=1.5,
        attack_range=0.75, move_speed=1.8, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["bats"] = CardDefinition(
        name="bats", display_name="Bats", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=1.0, hitpoints=81, damage=81, attack_speed=0.24,
        attack_range=0.75, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=5, spawned_unit="bat",
        target_air=True, target_ground=True,
    )
    registry["bat"] = CardDefinition(
        name="bat", display_name="Bat", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=81, damage=81, attack_speed=0.24,
        attack_range=0.75, move_speed=1.5, deployment_zone=DeploymentZone.EITHER_SIDE,
        target_air=True, target_ground=True,
    )

    # =========================================================================
    # 2 ELIXIR CARDS
    # =========================================================================
    registry["goblins"] = CardDefinition(
        name="goblins", display_name="Goblins", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=2.0, hitpoints=202, damage=120, attack_speed=0.23,
        attack_range=0.75, move_speed=1.8, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=4, spawned_unit="goblin",
    )
    registry["goblin"] = CardDefinition(
        name="goblin", display_name="Goblin", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=202, damage=120, attack_speed=0.23,
        attack_range=0.75, move_speed=1.8, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["spear_goblins"] = CardDefinition(
        name="spear_goblins", display_name="Spear Goblins", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=2.0, hitpoints=133, damage=81, attack_speed=0.71,
        attack_range=3.5, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="spear_goblin",
    )
    registry["spear_goblin"] = CardDefinition(
        name="spear_goblin", display_name="Spear Goblin", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=133, damage=81, attack_speed=0.71,
        attack_range=3.5, move_speed=1.5, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["bomber"] = CardDefinition(
        name="bomber", display_name="Bomber", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=2.0, hitpoints=304, damage=225, attack_speed=1.8,
        attack_range=3.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.0,
    )
    registry["ice_golem"] = CardDefinition(
        name="ice_golem", display_name="Ice Golem", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=2.0, hitpoints=1091, damage=69, attack_speed=1.5,
        attack_range=0.75, move_speed=1.4, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["suspicious_bush"] = CardDefinition(
        name="suspicious_bush", display_name="Suspicious Bush", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=2.0, hitpoints=67, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="goblin",
    )
    registry["barbarian_barrel"] = CardDefinition(
        name="barbarian_barrel", display_name="Barbarian Barrel", card_type="spell",
        cost=2.0, spell_damage=160, spell_radius=3.5,
        spell_duration=0.0, spell_type="damage",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["goblin_curse"] = CardDefinition(
        name="goblin_curse", display_name="Goblin Curse", card_type="spell",
        cost=2.0, spell_damage=160, spell_radius=4.5,
        spell_duration=3.0, spell_type="curse",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["wall_breakers"] = CardDefinition(
        name="wall_breakers", display_name="Wall Breakers", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=2.0, hitpoints=206, damage=244, attack_speed=0.5,
        attack_range=0.5, move_speed=2.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="wall_breaker_mini",
    )
    registry["wall_breaker_mini"] = CardDefinition(
        name="wall_breaker_mini", display_name="Wall Breaker", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=0.0, hitpoints=151, damage=244, attack_speed=0.5,
        attack_range=0.5, move_speed=2.5, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["zap"] = CardDefinition(
        name="zap", display_name="Zap", card_type="spell",
        cost=2.0, spell_damage=140, spell_radius=2.0,
        spell_duration=0.5, spell_type="stun",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["log"] = CardDefinition(
        name="log", display_name="The Log", card_type="spell",
        cost=2.0, spell_damage=380, spell_radius=3.5,
        spell_duration=0.0, spell_type="damage",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["rage"] = CardDefinition(
        name="rage", display_name="Rage", card_type="spell",
        cost=2.0, spell_damage=0, spell_radius=3.5,
        spell_duration=4.0, spell_type="rage",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["giant_snowball"] = CardDefinition(
        name="giant_snowball", display_name="Giant Snowball", card_type="spell",
        cost=2.0, spell_damage=192, spell_radius=4.5,
        spell_duration=0.0, spell_type="pushback",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["berserker"] = CardDefinition(
        name="berserker", display_name="Berserker", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=2.0, hitpoints=896, damage=102, attack_speed=0.6,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
    )

    # =========================================================================
    # 3 ELIXIR CARDS
    # =========================================================================
    registry["knight"] = CardDefinition(
        name="knight", display_name="Knight", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=1766, damage=202, attack_speed=1.2,
        attack_range=0.75, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["archers"] = CardDefinition(
        name="archers", display_name="Archers", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=304, damage=112, attack_speed=1.2,
        attack_range=4.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="archer",
    )
    registry["archer"] = CardDefinition(
        name="archer", display_name="Archer", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=304, damage=112, attack_speed=1.2,
        attack_range=4.5, move_speed=1.0, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["minions"] = CardDefinition(
        name="minions", display_name="Minions", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=230, damage=107, attack_speed=1.2,
        attack_range=1.0, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="minion",
        target_air=True, target_ground=True,
    )
    registry["minion"] = CardDefinition(
        name="minion", display_name="Minion", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=230, damage=107, attack_speed=1.2,
        attack_range=1.0, move_speed=1.5, deployment_zone=DeploymentZone.EITHER_SIDE,
        target_air=True, target_ground=True,
    )
    registry["barbarians"] = CardDefinition(
        name="barbarians", display_name="Barbarians", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=691, damage=192, attack_speed=0.28,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=5, spawned_unit="barbarian_mini",
    )
    registry["barbarian_mini"] = CardDefinition(
        name="barbarian_mini", display_name="Mini Barbarian", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=138, damage=192, attack_speed=0.28,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["skeleton_army"] = CardDefinition(
        name="skeleton_army", display_name="Skeleton Army", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=81, damage=81, attack_speed=0.37,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=8, spawned_unit="skeleton",
    )
    registry["valkyrie"] = CardDefinition(
        name="valkyrie", display_name="Valkyrie", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1579, damage=220, attack_speed=1.5,
        attack_range=1.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.0,
    )
    registry["goblin_gang"] = CardDefinition(
        name="goblin_gang", display_name="Goblin Gang", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=133, damage=81, attack_speed=0.71,
        attack_range=3.5, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="spear_goblin",
    )
    registry["princess"] = CardDefinition(
        name="princess", display_name="Princess", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=3.0, hitpoints=123, damage=79, attack_speed=1.2,
        attack_range=8.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["arrows"] = CardDefinition(
        name="arrows", display_name="Arrows", card_type="spell",
        cost=3.0, spell_damage=268, spell_radius=3.0,
        spell_duration=0.0, spell_type="damage",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["tombstone"] = CardDefinition(
        name="tombstone", display_name="Tombstone", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=438, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="skeleton",
    )
    registry["cannon"] = CardDefinition(
        name="cannon", display_name="Cannon", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=824, damage=202, attack_speed=1.1,
        attack_range=3.5, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["earthquake"] = CardDefinition(
        name="earthquake", display_name="Earthquake", card_type="spell",
        cost=3.0, spell_damage=180, spell_radius=4.0,
        spell_duration=4.0, spell_type="earthquake",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["vines"] = CardDefinition(
        name="vines", display_name="Vines", card_type="spell",
        cost=3.0, spell_damage=192, spell_radius=4.5,
        spell_duration=3.0, spell_type="root",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["clone"] = CardDefinition(
        name="clone", display_name="Clone", card_type="spell",
        cost=3.0, spell_damage=0, spell_radius=3.5,
        spell_duration=8.0, spell_type="clone",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["goblin_barrel"] = CardDefinition(
        name="goblin_barrel", display_name="Goblin Barrel", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=3.0, hitpoints=402, damage=151, attack_speed=1.0,
        attack_range=0.5, move_speed=2.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="goblin",
    )
    registry["skeleton_barrel"] = CardDefinition(
        name="skeleton_barrel", display_name="Skeleton Barrel", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=532, damage=201, attack_speed=1.0,
        attack_range=0.5, move_speed=2.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="skeleton",
    )
    registry["firecracker"] = CardDefinition(
        name="firecracker", display_name="Firecracker", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=304, damage=320, attack_speed=1.8,
        attack_range=5.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.0,
    )
    registry["royal_delivery"] = CardDefinition(
        name="royal_delivery", display_name="Royal Delivery", card_type="spell",
        cost=3.0, spell_damage=0, spell_radius=4.5,
        spell_duration=0.0, spell_type="deliver", spawn_count=2,
        spawned_unit="barbarian_mini", deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["miner"] = CardDefinition(
        name="miner", display_name="Miner", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=572, damage=91, attack_speed=1.1,
        attack_range=0.75, move_speed=1.4, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["ice_wizard"] = CardDefinition(
        name="ice_wizard", display_name="Ice Wizard", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=325, damage=42, attack_speed=1.75,
        attack_range=4.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["guards"] = CardDefinition(
        name="guards", display_name="Guards", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=51, damage=73, attack_speed=0.24,
        attack_range=0.75, move_speed=1.6, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="guard",
    )
    registry["guard"] = CardDefinition(
        name="guard", display_name="Guard", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=51, damage=73, attack_speed=0.24,
        attack_range=0.75, move_speed=1.6, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["dart_goblin"] = CardDefinition(
        name="dart_goblin", display_name="Dart Goblin", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=261, damage=151, attack_speed=0.8,
        attack_range=6.5, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["fisherman"] = CardDefinition(
        name="fisherman", display_name="Fisherman", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=411, damage=91, attack_speed=1.4,
        attack_range=2.5, move_speed=1.4, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["little_prince"] = CardDefinition(
        name="little_prince", display_name="Little Prince", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=3.0, hitpoints=273, damage=41, attack_speed=1.2,
        attack_range=5.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["bandit"] = CardDefinition(
        name="bandit", display_name="Bandit", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=428, damage=91, attack_speed=1.0,
        attack_range=0.75, move_speed=1.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["royal_ghost"] = CardDefinition(
        name="royal_ghost", display_name="Royal Ghost", card_type="unit",
        # A ground troop that turns invisible, not an air unit -- classing it
        # as AIR made it immune to every ground-only attacker.
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=3.0, hitpoints=572, damage=123, attack_speed=1.2,
        attack_range=1.0, move_speed=2.0, deployment_zone=DeploymentZone.EITHER_SIDE,
        target_air=True, target_ground=True,
    )

    # =========================================================================
    # 4 ELIXIR CARDS (and misc cost cards)
    # =========================================================================
    registry["musketeer"] = CardDefinition(
        name="musketeer", display_name="Musketeer", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=597, damage=180, attack_speed=1.2,
        attack_range=5.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["fireball"] = CardDefinition(
        name="fireball", display_name="Fireball", card_type="spell",
        cost=4.0, spell_damage=688, spell_radius=2.5,
        spell_duration=0.0, spell_type="damage",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["wizard"] = CardDefinition(
        name="wizard", display_name="Wizard", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=625, damage=233, attack_speed=1.4,
        attack_range=4.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.5,
    )
    registry["baby_dragon"] = CardDefinition(
        name="baby_dragon", display_name="Baby Dragon", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=720, damage=100, attack_speed=1.5,
        attack_range=3.0, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.0, target_air=True, target_ground=True,
    )
    registry["mega_minion"] = CardDefinition(
        name="mega_minion", display_name="Mega Minion", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=693, damage=258, attack_speed=1.5,
        attack_range=1.0, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
    )
    registry["mini_pekka"] = CardDefinition(
        name="mini_pekka", display_name="Mini P.E.K.K.A", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1153, damage=625, attack_speed=1.6,
        attack_range=0.75, move_speed=1.4, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["hog_rider"] = CardDefinition(
        name="hog_rider", display_name="Hog Rider", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=4.0, hitpoints=1405, damage=262, attack_speed=1.5,
        attack_range=1.0, move_speed=2.5, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["electro_wizard"] = CardDefinition(
        name="electro_wizard", display_name="Electro Wizard", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=337, damage=111, attack_speed=1.2,
        attack_range=3.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["lumberjack"] = CardDefinition(
        name="lumberjack", display_name="Lumberjack", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=606, damage=121, attack_speed=1.2,
        attack_range=0.75, move_speed=1.8, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["electro_dragon"] = CardDefinition(
        name="electro_dragon", display_name="Electro Dragon", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=624, damage=120, attack_speed=1.2,
        attack_range=1.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["dragon"] = CardDefinition(
        name="dragon", display_name="Dragon", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1262, damage=122, attack_speed=1.5,
        attack_range=1.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.0, target_air=True, target_ground=True,
    )
    registry["dark_prince"] = CardDefinition(
        name="dark_prince", display_name="Dark Prince", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=750, damage=166, attack_speed=1.4,
        attack_range=0.75, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["executioner"] = CardDefinition(
        name="executioner", display_name="Executioner", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=800, damage=112, attack_speed=2.4,
        attack_range=5.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.5,
    )
    registry["giant_skeleton"] = CardDefinition(
        name="giant_skeleton", display_name="Giant Skeleton", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=6.0, hitpoints=2100, damage=172, attack_speed=1.5,
        attack_range=0.75, move_speed=0.7, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["royal_hogs"] = CardDefinition(
        name="royal_hogs", display_name="Royal Hogs", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=693, damage=61, attack_speed=0.75,
        attack_range=2.0, move_speed=2.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="royal_hog",
    )
    registry["royal_hog"] = CardDefinition(
        name="royal_hog", display_name="Royal Hog", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=0.0, hitpoints=693, damage=61, attack_speed=0.75,
        attack_range=2.0, move_speed=2.0, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["inferno_dragon"] = CardDefinition(
        name="inferno_dragon", display_name="Inferno Dragon", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=612, damage=97, attack_speed=0.85,
        attack_range=3.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
    )
    registry["battle_ram"] = CardDefinition(
        name="battle_ram", display_name="Battle Ram", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=801, damage=237, attack_speed=1.5,
        attack_range=2.5, move_speed=1.6, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="barbarian_mini",
    )
    registry["zappies"] = CardDefinition(
        name="zappies", display_name="Zappies", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=438, damage=97, attack_speed=2.1,
        attack_range=5.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="zappy", target_air=True, target_ground=True,
    )
    registry["zappy"] = CardDefinition(
        name="zappy", display_name="Zappy", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=81, damage=97, attack_speed=2.1,
        attack_range=5.0, move_speed=1.5, deployment_zone=DeploymentZone.EITHER_SIDE,
        target_air=True, target_ground=True,
    )
    registry["flying_machine"] = CardDefinition(
        name="flying_machine", display_name="Flying Machine", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=4.0, hitpoints=508, damage=142, attack_speed=1.5,
        attack_range=5.0, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
    )
    registry["river_rider"] = CardDefinition(
        name="river_rider", display_name="River Rider", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=2000, damage=150, attack_speed=1.5,
        attack_range=0.75, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["battle_healer"] = CardDefinition(
        name="battle_healer", display_name="Battle Healer", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1422, damage=122, attack_speed=1.5,
        attack_range=3.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["goblin_demolisher"] = CardDefinition(
        name="goblin_demolisher", display_name="Goblin Demolisher", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=4.0, hitpoints=1076, damage=154, attack_speed=1.2,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["rune_giant"] = CardDefinition(
        name="rune_giant", display_name="Rune Giant", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=1664, damage=75, attack_speed=1.5,
        attack_range=0.75, move_speed=0.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["phoenix"] = CardDefinition(
        name="phoenix", display_name="Phoenix", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=497, damage=102, attack_speed=1.0,
        attack_range=3.0, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.5, target_air=True, target_ground=True,
    )
    registry["ronin"] = CardDefinition(
        name="ronin", display_name="Ronin", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=750, damage=120, attack_speed=1.2,
        attack_range=3.0, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["magic_archer"] = CardDefinition(
        name="magic_archer", display_name="Magic Archer", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=250, damage=67, attack_speed=1.1,
        attack_range=5.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["mother_witch"] = CardDefinition(
        name="mother_witch", display_name="Mother Witch", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=250, damage=62, attack_speed=1.0,
        attack_range=3.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["mighty_miner"] = CardDefinition(
        name="mighty_miner", display_name="Mighty Miner", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=879, damage=16, attack_speed=0.4,
        attack_range=0.75, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["skeleton_king"] = CardDefinition(
        name="skeleton_king", display_name="Skeleton King", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=898, damage=80, attack_speed=1.6,
        attack_range=0.75, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["golden_knight"] = CardDefinition(
        name="golden_knight", display_name="Golden Knight", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=703, damage=63, attack_speed=0.9,
        attack_range=2.5, move_speed=1.4, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["monk"] = CardDefinition(
        name="monk", display_name="Monk", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=865, damage=55, attack_speed=0.8,
        attack_range=3.5, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["archer_queen"] = CardDefinition(
        name="archer_queen", display_name="Archer Queen", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=391, damage=88, attack_speed=1.2,
        attack_range=6.0, move_speed=1.3, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["goblin_machine"] = CardDefinition(
        name="goblin_machine", display_name="Goblin Machine", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=1016, damage=100, attack_speed=1.2,
        attack_range=3.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
    )
    registry["goblinstein"] = CardDefinition(
        name="goblinstein", display_name="Goblinstein", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=1800, damage=120, attack_speed=1.5,
        attack_range=0.75, move_speed=0.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["boss_bandit"] = CardDefinition(
        name="boss_bandit", display_name="Boss Bandit", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=6.0, hitpoints=1025, damage=96, attack_speed=1.1,
        attack_range=3.0, move_speed=1.4, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["elixir_storage"] = CardDefinition(
        name="elixir_storage", display_name="Elixir Storage", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1200, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        elixir_rate=0.0,
    )
    registry["tesla"] = CardDefinition(
        name="tesla", display_name="Tesla", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1152, damage=220, attack_speed=1.1,
        attack_range=3.5, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["mortar"] = CardDefinition(
        name="mortar", display_name="Mortar", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1369, damage=266, attack_speed=3.4,
        attack_range=7.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.0, target_air=False,
    )
    registry["bomb_tower"] = CardDefinition(
        name="bomb_tower", display_name="Bomb Tower", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=1123, damage=184, attack_speed=1.6,
        attack_range=4.5, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        aoe=True, aoe_radius=1.5,
    )
    registry["furnace"] = CardDefinition(
        name="furnace", display_name="Furnace", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=742, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        elixir_rate=4.0, spawn_count=1, spawned_unit="fire_spirit",
    )
    registry["goblin_cage"] = CardDefinition(
        name="goblin_cage", display_name="Goblin Cage", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=646, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=4, spawned_unit="goblin",
    )
    registry["goblin_drill"] = CardDefinition(
        name="goblin_drill", display_name="Goblin Drill", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.PATH_FORWARD,
        cost=4.0, hitpoints=820, damage=52, attack_speed=0.5,
        attack_range=0.75, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=1, spawned_unit="skeleton",
    )
    registry["goblin_hut"] = CardDefinition(
        name="goblin_hut", display_name="Goblin Hut", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=977, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        elixir_rate=3.0, spawn_count=1, spawned_unit="goblin",
    )

    # =========================================================================
    # 5 ELIXIR CARDS (and misc cost cards)
    # =========================================================================
    registry["poison"] = CardDefinition(
        name="poison", display_name="Poison", card_type="spell",
        cost=4.0, spell_damage=368, spell_radius=3.0,
        spell_duration=2.75, spell_type="poison",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["freeze"] = CardDefinition(
        name="freeze", display_name="Freeze", card_type="spell",
        cost=4.0, spell_damage=0, spell_radius=3.5,
        spell_duration=1.8, spell_type="freeze",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["heal_spell"] = CardDefinition(
        name="heal_spell", display_name="Heal Spell", card_type="spell",
        cost=4.0, spell_damage=0, spell_radius=3.0,
        spell_duration=2.0, spell_type="heal",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["giant"] = CardDefinition(
        name="giant", display_name="Giant", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=3286, damage=209, attack_speed=1.5,
        attack_range=0.75, move_speed=0.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["balloon"] = CardDefinition(
        name="balloon", display_name="Balloon", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=1048, damage=400, attack_speed=2.0,
        attack_range=1.5, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
    )
    registry["witch"] = CardDefinition(
        name="witch", display_name="Witch", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=524, damage=84, attack_speed=1.2,
        attack_range=3.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="skeleton",
    )
    registry["night_witch"] = CardDefinition(
        name="night_witch", display_name="Night Witch", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=428, damage=106, attack_speed=1.2,
        attack_range=3.5, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=1, spawned_unit="bat",
    )
    registry["minion_horde"] = CardDefinition(
        name="minion_horde", display_name="Minion Horde", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=202, damage=61, attack_speed=1.2,
        attack_range=1.0, move_speed=1.5, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=6, spawned_unit="minion",
        target_air=True, target_ground=True,
    )
    registry["prince"] = CardDefinition(
        name="prince", display_name="Prince", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=5.0, hitpoints=1200, damage=384, attack_speed=1.4,
        attack_range=0.75, move_speed=1.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["prince_mini"] = CardDefinition(
        name="prince_mini", display_name="Prince (Post-Charge)", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=1200, damage=192, attack_speed=1.4,
        attack_range=0.75, move_speed=1.6, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["ram_rider"] = CardDefinition(
        name="ram_rider", display_name="Ram Rider", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=802, damage=118, attack_speed=1.5,
        attack_range=2.5, move_speed=1.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["elite_barbarians"] = CardDefinition(
        name="elite_barbarians", display_name="Elite Barbarians", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST,
        cost=6.0, hitpoints=1341, damage=384, attack_speed=0.7,
        attack_range=0.75, move_speed=2.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="elite_barbarian_mini",
    )
    registry["elite_barbarian_mini"] = CardDefinition(
        name="elite_barbarian_mini", display_name="Elite Barbarian", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=671, damage=384, attack_speed=0.7,
        attack_range=0.75, move_speed=2.0, deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["skeleton_dragon"] = CardDefinition(
        name="skeleton_dragon", display_name="Skeleton Dragon", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=560, damage=151, attack_speed=1.0,
        attack_range=3.0, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
    )
    registry["skeleton_dragons"] = CardDefinition(
        name="skeleton_dragons", display_name="Skeleton Dragons", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST,
        cost=4.0, hitpoints=560, damage=151, attack_speed=1.0,
        attack_range=3.0, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=2, spawned_unit="skeleton_dragon",
        target_air=True, target_ground=True,
    )
    registry["rascals"] = CardDefinition(
        name="rascals", display_name="Rascals", card_type="unit",
        unit_type=UnitType.SWARM, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=5.0, hitpoints=800, damage=120, attack_speed=1.0,
        attack_range=3.5, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="rascal",
    )
    registry["rascal"] = CardDefinition(
        name="rascal", display_name="Rascal", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=0.0, hitpoints=267, damage=120, attack_speed=1.0,
        attack_range=3.5, move_speed=1.2, deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    # =========================================================================
    # 6 ELIXIR CARDS
    # =========================================================================
    registry["lightning"] = CardDefinition(
        name="lightning", display_name="Lightning", card_type="spell",
        cost=6.0, spell_damage=820, spell_radius=1.5,
        spell_type="damage", deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["rocket"] = CardDefinition(
        name="rocket", display_name="Rocket", card_type="spell",
        cost=6.0, spell_damage=1484, spell_radius=2.5,
        spell_duration=0.0, spell_type="damage",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["elixir_collector"] = CardDefinition(
        name="elixir_collector", display_name="Elixir Collector", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=6.0, hitpoints=886, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        elixir_rate=2.45,
    )
    registry["x_bow"] = CardDefinition(
        name="x_bow", display_name="X-Bow", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=6.0, hitpoints=1000, damage=27, attack_speed=0.3,
        attack_range=6.5, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=False,
    )
    registry["sparky"] = CardDefinition(
        name="sparky", display_name="Sparky", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=6.0, hitpoints=686, damage=629, attack_speed=1.6,
        attack_range=5.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["goblin_giant"] = CardDefinition(
        name="goblin_giant", display_name="Goblin Giant", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=6.0, hitpoints=1888, damage=110, attack_speed=1.5,
        attack_range=0.75, move_speed=0.6, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["royal_giant"] = CardDefinition(
        name="royal_giant", display_name="Royal Giant", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=6.0, hitpoints=3164, damage=307, attack_speed=1.5,
        attack_range=5.0, move_speed=0.5, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["spirit_empress"] = CardDefinition(
        name="spirit_empress", display_name="Spirit Empress", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=6.0, hitpoints=529, damage=145, attack_speed=1.4,
        attack_range=5.0, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )

    # =========================================================================
    # 7 ELIXIR CARDS
    # =========================================================================
    registry["mega_knight"] = CardDefinition(
        name="mega_knight", display_name="Mega Knight", card_type="unit",
        unit_type=UnitType.MELEE, targeting_mode=TargetingMode.NEAREST,
        cost=7.0, hitpoints=1887, damage=127, attack_speed=1.2,
        attack_range=0.75, move_speed=1.2, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["lava_hound"] = CardDefinition(
        name="lava_hound", display_name="Lava Hound", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=7.0, hitpoints=1692, damage=25, attack_speed=1.5,
        attack_range=1.0, move_speed=0.8, deployment_zone=DeploymentZone.SELF_SIDE,
        target_air=True, target_ground=True,
        death_spawn_count=6, death_spawned_unit="lava_pup",
    )
    registry["lava_pup"] = CardDefinition(
        name="lava_pup", display_name="Lava Pup", card_type="unit",
        unit_type=UnitType.AIR, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=0.0, hitpoints=300, damage=60, attack_speed=1.5,
        attack_range=1.0, move_speed=1.5, deployment_zone=DeploymentZone.EITHER_SIDE,
        target_air=True, target_ground=True,
    )
    registry["royal_recruit"] = CardDefinition(
        name="royal_recruit", display_name="Royal Recruit", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=7.0, hitpoints=547, damage=133, attack_speed=2.08,
        attack_range=0.75, move_speed=0.7, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["barbarian_hut"] = CardDefinition(
        name="barbarian_hut", display_name="Barbarian Hut", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=7.0, hitpoints=964, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
        elixir_rate=3.0, spawn_count=2, spawned_unit="barbarian_mini",
    )

    # =========================================================================
    # 8 ELIXIR CARDS
    # =========================================================================
    registry["golem"] = CardDefinition(
        name="golem", display_name="Golem", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=8.0, hitpoints=3200, damage=195, attack_speed=1.6,
        attack_range=0.75, move_speed=0.5, deployment_zone=DeploymentZone.SELF_SIDE,
        death_spawn_count=2, death_spawned_unit="golem_mini",
    )
    registry["golem_mini"] = CardDefinition(
        name="golem_mini", display_name="Golem Jr.", card_type="unit",
        unit_type=UnitType.TANK, targeting_mode=TargetingMode.NEAREST_TOWER,
        cost=0.0, hitpoints=1600, damage=86, attack_speed=1.6,
        attack_range=0.75, move_speed=0.5, deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    # =========================================================================
    # 9 ELIXIR CARDS
    # =========================================================================
    registry["three_musketeers"] = CardDefinition(
        name="three_musketeers", display_name="Three Musketeers", card_type="unit",
        unit_type=UnitType.RANGED, targeting_mode=TargetingMode.NEAREST,
        cost=9.0, hitpoints=731, damage=169, attack_speed=1.2,
        attack_range=5.5, move_speed=1.0, deployment_zone=DeploymentZone.SELF_SIDE,
        spawn_count=3, spawned_unit="musketeer",
    )

    # =========================================================================
    # FREE / SPECIAL CARDS (cost 0)
    # =========================================================================
    registry["elixir_spring"] = CardDefinition(
        name="elixir_spring", display_name="Elixir Spring", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=0.0, hitpoints=1, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.EITHER_SIDE,
        elixir_rate=1.0,
    )
    registry["graveyard"] = CardDefinition(
        name="graveyard", display_name="Graveyard", card_type="spell",
        cost=5.0, spell_damage=0, spell_radius=2.5,
        spell_duration=0.0, spell_type="graveyard",
        spawn_count=8, spawned_unit="skeleton",
        deployment_zone=DeploymentZone.OPPONENT_SIDE,
    )
    registry["tornado"] = CardDefinition(
        name="tornado", display_name="Tornado", card_type="spell",
        cost=3.0, spell_damage=0, spell_radius=3.5,
        spell_duration=3.0, spell_type="tornado",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["mirror"] = CardDefinition(
        name="mirror", display_name="Mirror", card_type="spell",
        cost=0.0, spell_damage=0, spell_radius=3.5,
        spell_duration=0.0, spell_type="mirror",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["clash_ferry"] = CardDefinition(
        name="clash_ferry", display_name="Clash Ferry", card_type="unit",
        unit_type=UnitType.BUILDING, targeting_mode=TargetingMode.NEAREST,
        cost=3.0, hitpoints=1200, damage=0, attack_speed=0.0,
        attack_range=0.0, move_speed=0.0, deployment_zone=DeploymentZone.SELF_SIDE,
    )
    registry["royal_deliver"] = CardDefinition(
        name="royal_deliver", display_name="Royal Deliver", card_type="spell",
        cost=4.0, spell_damage=0, spell_radius=3.0,
        spell_type="deliver", spawn_count=1, spawned_unit="royal_ghost",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )
    registry["void"] = CardDefinition(
        name="void", display_name="Void", card_type="spell",
        cost=5.0, spell_damage=268, spell_radius=4.5,
        spell_duration=3.0, spell_type="void",
        deployment_zone=DeploymentZone.EITHER_SIDE,
    )

    return registry



def get_default_tower_registry() -> dict[str, TowerDefinition]:
    """Return default tower definitions for both players."""
    towers: dict[str, TowerDefinition] = {}

    # Princess towers (left, right) - shared definition.
    # Level 11 stats, matching the card registry. The previous values (1400 HP
    # / 70 damage / 1.2s) left towers so far below card power that a single
    # mid-cost troop could solo one, so every match ended in a fast king rush
    # instead of ever reaching time.
    princess_tower = TowerDefinition(
        name="princess_tower",
        display_name="Princess Tower",
        hitpoints=2534,
        damage=109,
        attack_speed=0.8,
        attack_range=7.5,
        crown_reward=1,
        is_king_tower=False,
    )

    king_tower = TowerDefinition(
        name="king_tower",
        display_name="King Tower",
        hitpoints=4008,
        damage=109,
        attack_speed=1.0,
        attack_range=7.0,
        crown_reward=3,
        is_king_tower=True,
        is_active=False,
    )

    # Player 1 towers (bottom side)
    _p1_princess = {k: v for k, v in princess_tower.to_dict().items() if k not in ("name", "x", "y")}
    towers["p1_princess_left"] = TowerDefinition(**_p1_princess, name="p1_princess_left", x=-3.0, y=8.0)
    towers["p1_princess_right"] = TowerDefinition(**_p1_princess, name="p1_princess_right", x=3.0, y=8.0)
    _p1_king = {k: v for k, v in king_tower.to_dict().items() if k not in ("name", "x", "y")}
    towers["p1_king"] = TowerDefinition(**_p1_king, name="p1_king", x=0.0, y=10.0)

    # Player 2 towers (top side)
    _p2_princess = {k: v for k, v in princess_tower.to_dict().items() if k not in ("name", "x", "y")}
    towers["p2_princess_left"] = TowerDefinition(**_p2_princess, name="p2_princess_left", x=-3.0, y=-8.0)
    towers["p2_princess_right"] = TowerDefinition(**_p2_princess, name="p2_princess_right", x=3.0, y=-8.0)
    _p2_king = {k: v for k, v in king_tower.to_dict().items() if k not in ("name", "x", "y")}
    towers["p2_king"] = TowerDefinition(**_p2_king, name="p2_king", x=0.0, y=-10.0)

    return towers

CARD_DEFS = get_default_card_registry()
TOWER_DEFS = get_default_tower_registry()
