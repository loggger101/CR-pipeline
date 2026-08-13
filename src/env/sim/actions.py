"""
CR-Pipeline: Simulation Action Space

Defines the action space for evolutionary agents playing Clash Royale.
Actions are decomposed into card selection + placement, with validation
against game rules (elixir, cooldowns, deployment zones).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class ActionType(enum.Enum):
    """Base types of actions an agent can take."""
    # Card deployment actions
    DEPLOY_UNIT = enum.auto()      # Deploy a unit card
    DEPLOY_SPELL = enum.auto()     # Cast a spell card
    # Pass / no-op
    PASS = enum.auto()             # Do nothing this tick
    # Future extensions:
    # SWAP_CARD = enum.auto()       # Swap position in hand
    # TARGET_ENEMY = enum.auto()    # Target specific enemy (for precision spells)


@dataclass(frozen=True)
class Action:
    """
    A single action from the agent.

    For DEPLOY_UNIT / DEPLOY_SPELL:
        card_index: Index into the agent's current hand (0-3)
        target_x: X coordinate on the arena (float)
        target_y: Y coordinate on the arena (float)
        target_id: Optional target entity ID (for targeted spells)

    For PASS:
        All fields are None.

    Attributes:
        action_type: The type of action.
        card_index: Which card from the hand (0-3), or None for PASS.
        target_x: X position for deployment / spell center.
        target_y: Y position for deployment / spell center.
        target_id: Optional target entity ID.
        raw_output: Raw neural network output for logging.
    """
    action_type: ActionType
    card_index: Optional[int] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_id: Optional[str] = None
    raw_output: Optional[list[float]] = None

    def is_valid_action(self) -> bool:
        """Check if the action has required fields for its type."""
        if self.action_type == ActionType.PASS:
            return True
        if self.action_type in (ActionType.DEPLOY_UNIT, ActionType.DEPLOY_SPELL):
            if self.card_index is None:
                return False
            if self.target_x is None or self.target_y is None:
                return False
            return True
        return False


class ActionSpace:
    """
    Defines and validates the action space for a single agent.

    The action space is decomposed:
        - Card selection: discrete index into hand (0-3) + PASS
        - Placement: continuous 2D position clipped to valid deployment zone

    For neural network output, the agent produces:
        - card_logits: float32[5] (4 cards + PASS)
        - position: float32[2] (normalized arena coordinates)

    Total output dimension: 7 (configurable).
    """

    def __init__(
        self,
        hand_size: int = 4,
        arena_width: float = 8.0,
        arena_height: float = 18.0,
        use_discrete_position: bool = False,
        grid_size: int = 10,
    ):
        """
        Args:
            hand_size: Number of cards in the agent's hand.
            arena_width: Width of the arena in simulation units.
            arena_height: Height of the arena in simulation units.
            use_discrete_position: If True, discretize placement to a grid.
            grid_size: Number of grid cells per axis (if discrete).
        """
        self.hand_size = hand_size
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.use_discrete_position = use_discrete_position
        self.grid_size = grid_size

        # Total output dimension from neural network
        # card_logits: hand_size + 1 (for PASS)
        # position: 2 (x, y)
        self.output_dim = (hand_size + 1) + 2

    def parse_network_output(self, output: list[float]) -> Action:
        """
        Parse raw neural network output into an Action.

        Args:
            output: Raw output array of length self.output_dim.
                    First (hand_size + 1) values are card logits.
                    Last 2 values are continuous position.

        Returns:
            Parsed Action object.
        """
        if len(output) != self.output_dim:
            raise ValueError(
                f"Expected output dim {self.output_dim}, got {len(output)}"
            )

        card_logits = output[: self.hand_size + 1]
        position = output[self.hand_size + 1 :]

        # Select card: argmax of logits
        best_card_idx = card_logits.index(max(card_logits))

        if best_card_idx >= self.hand_size:
            # PASS was selected
            return Action(
                action_type=ActionType.PASS,
                raw_output=output,
            )

        # Continuous position -> arena coordinates
        nx, ny = position[0], position[1]

        # Normalize from [0, 1] to arena coordinates
        tx = (nx - 0.5) * self.arena_width
        ty = (ny - 0.5) * self.arena_height

        if self.use_discrete_position:
            # Quantize to grid
            grid_x = int(tx * self.grid_size) / self.grid_size
            grid_y = int(ty * self.grid_size) / self.grid_size
            tx = grid_x
            ty = grid_y

        return Action(
            action_type=ActionType.DEPLOY_UNIT,
            card_index=best_card_idx,
            target_x=tx,
            target_y=ty,
            raw_output=output,
        )

    def clip_position_to_zone(
        self,
        x: float,
        y: float,
        owner: int,
    ) -> tuple[float, float]:
        """
        Clip a target position to the valid deployment zone for a player.

        Args:
            x: X coordinate.
            y: Y coordinate.
            owner: Player ID (0 = bottom, 1 = top).

        Returns:
            Clipped (x, y) coordinates.
        """
        # Arena bounds
        x = max(-self.arena_width / 2, min(self.arena_width / 2, x))

        if owner == 0:
            # Player 1 deploys on bottom half (y > 0)
            y = max(0.0, min(self.arena_height / 2, y))
        else:
            # Player 2 deploys on top half (y < 0)
            y = max(-self.arena_height / 2, min(0.0, y))

        return x, y

    def get_action_mask(
        self,
        hand: list,
        elixir: list[float],
    ) -> list[bool]:
        """
        Generate action mask indicating which cards can be played.

        Args:
            hand: List of card definitions currently in hand.
            elixir: Current elixir level for each card.

        Returns:
            Boolean mask of length (hand_size + 1). True = playable.
            Last entry is always True (PASS is always available).
        """
        mask = []
        for i, card in enumerate(hand):
            if i < len(elixir):
                mask.append(elixir[i] >= card.cost)
            else:
                mask.append(False)
        mask.append(True)  # PASS always available
        return mask

    def to_dict(self) -> dict:
        """Serialize action space config."""
        return {
            "hand_size": self.hand_size,
            "arena_width": self.arena_width,
            "arena_height": self.arena_height,
            "use_discrete_position": self.use_discrete_position,
            "grid_size": self.grid_size,
            "output_dim": self.output_dim,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ActionSpace:
        """Deserialize action space config."""
        return cls(
            hand_size=data["hand_size"],
            arena_width=data["arena_width"],
            arena_height=data["arena_height"],
            use_discrete_position=data.get("use_discrete_position", False),
            grid_size=data.get("grid_size", 10),
        )


class ActionValidator:
    """
    Validates whether an Action can be legally executed given the current
    game state (elixir, cooldowns, deployment zone, etc.).
    """

    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space

    def validate(
        self,
        action: Action,
        hand: list,
        elixir_levels: list[float],
        cooldowns: list[float],
        current_player: int,
    ) -> tuple[bool, str]:
        """
        Validate an action against game rules.

        Args:
            action: The action to validate.
            hand: Current card hand.
            elixir_levels: Current elixir for each card.
            cooldowns: Current cooldowns for each card (seconds remaining).
            current_player: Player ID (0 or 1).

        Returns:
            (is_valid, reason): Tuple of validity and explanation string.
        """
        if action.action_type == ActionType.PASS:
            return True, "PASS"

        if action.card_index is None or action.card_index < 0:
            return False, "Invalid card index"

        if action.card_index >= len(hand):
            return False, "Card index out of range"

        card = hand[action.card_index]

        # Check elixir
        if elixir_levels[action.card_index] < card.cost:
            return False, f"Insufficient elixir for {card.name}"

        # Check cooldown
        if cooldowns[action.card_index] > 0:
            return False, f"Card {card.name} on cooldown"

        # Check deployment zone
        if card.deployment_zone is not None:
            if (
                card.deployment_zone == DeploymentZone.SELF_SIDE
                and current_player == 1
            ):
                return False, "Cannot deploy on opponent side"
            if (
                card.deployment_zone == DeploymentZone.OPPONENT_SIDE
                and current_player == 0
            ):
                return False, "Cannot deploy on self side"

        return True, "Valid"

    def execute_if_valid(
        self,
        action: Action,
        hand: list,
        elixir_levels: list[float],
        cooldowns: list[float],
        current_player: int,
    ) -> tuple[bool, Action | None, str]:
        """
        Validate and execute an action.

        Returns:
            (success, executed_action, reason)
        """
        valid, reason = self.validate(action, hand, elixir_levels, cooldowns, current_player)
        if not valid:
            return False, None, reason

        # Deduct elixir
        if action.card_index is not None:
            elixir_levels[action.card_index] -= hand[action.card_index].cost

        return True, action, reason
