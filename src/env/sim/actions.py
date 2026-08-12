"""Discrete action space for the simulation environment.

Actions are encoded as (card_index, target_col, target_row) tuples where
card_index selects from the agent's hand, and (col, row) specifies the
deployment cell on the arena grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np


class ActionType(Enum):
    """Types of actions the agent can take."""
    PLAY_CARD = auto()       # Deploy a card at a grid cell
    PASS = auto()            # Skip this turn
    INVALID = auto()         # Placeholder for invalid/filtered actions


@dataclass(frozen=True)
class Action:
    """A single action in the simulation.

    Attributes:
        action_type: Type of action.
        card_idx: Index into the agent's card hand (0-3).
        target_col: Grid column of the target cell.
        target_row: Grid row of the target cell.
        valid: Whether this action is legal given current game state.
    """
    action_type: ActionType = ActionType.PASS
    card_idx: int = -1
    target_col: float = -1.0
    target_row: float = -1.0
    valid: bool = False

    @classmethod
    def play_card(cls, card_idx: int, target_col: float, target_row: float) -> Action:
        """Create a PLAY_CARD action."""
        return cls(
            action_type=ActionType.PLAY_CARD,
            card_idx=card_idx,
            target_col=target_col,
            target_row=target_row,
            valid=False,  # Will be set by validator
        )

    @classmethod
    def pass_action(cls) -> Action:
        """Create a PASS action."""
        return cls(action_type=ActionType.PASS, valid=True)

    @classmethod
    def invalid(cls) -> Action:
        """Create an INVALID action."""
        return cls(action_type=ActionType.INVALID, valid=False)

    def to_tuple(self) -> Tuple[int, float, float]:
        """Encode as (card_idx, target_col, target_row)."""
        return (self.card_idx, self.target_col, self.target_row)

    def to_array(self) -> np.ndarray:
        """Encode as a flat numpy array for neural net input.

        Returns:
            Array of shape (3,) with [card_idx, target_col, target_row].
            card_idx is -1 for PASS.
        """
        return np.array([self.card_idx, self.target_col, self.target_row],
                        dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> Action:
        """Decode an action from a numpy array."""
        card_idx = int(arr[0])
        target_col = float(arr[1])
        target_row = float(arr[2])
        if card_idx == -1:
            return Action.pass_action()
        return Action.play_card(card_idx, target_col, target_row)


class ActionSpace:
    """Defines the structure of the action space.

    The action space is decomposed:
      - Card selection: 0-3 (hand indices) or -1 (pass)
      - Placement: continuous (col, row) in grid coordinates

    This allows the neural net to output continuous values for placement
    while keeping card selection discrete (argmax over 5 choices).
    """

    def __init__(self, hand_size: int = 4, grid_cols: int = 8, grid_rows: int = 6):
        self.hand_size = hand_size
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        # Total discrete actions: (hand_size + 1) * grid_cols * grid_rows + 1 (pass)
        self.total_actions = (hand_size + 1) * grid_cols * grid_rows + 1

    def is_valid_placement(self, action: Action, arena: np.ndarray) -> bool:
        """Check if the action's placement is within the arena bounds."""
        if action.action_type != ActionType.PLAY_CARD:
            return True
        col = action.target_col
        row = action.target_row
        if not (0 <= col < self.grid_cols and 0 <= row < self.grid_rows):
            return False
        # Check the cell isn't occupied by a tower
        if arena[int(row), int(col)] != 0:
            return False
        return True

    def clip_to_arena(self, action: Action) -> Action:
        """Clip action placement to valid arena bounds."""
        if action.action_type != ActionType.PLAY_CARD:
            return action
        col = np.clip(action.target_col, 0, self.grid_cols - 1)
        row = np.clip(action.target_row, 0, self.grid_rows - 1)
        return Action(
            action_type=ActionType.PLAY_CARD,
            card_idx=action.card_idx,
            target_col=float(col),
            target_row=float(row),
            valid=True,
        )


class ActionValidator:
    """Validates actions against game state constraints.

    Checks:
    - Card is in agent's hand
    - Card is not on cooldown
    - Agent has enough elixir
    - Placement is in valid deployment zone for the card type
    - Placement cell is not occupied
    """

    def __init__(self, grid_cols: int = 8, grid_rows: int = 6):
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

    def validate(self, action: Action, hand: list, cooldowns: np.ndarray,
                 elixir: float, card_defs: dict, arena: np.ndarray,
                 is_player_side: bool = True) -> bool:
        """Validate an action against current game state.

        Args:
            action: The action to validate.
            hand: List of card names currently in hand.
            cooldowns: Array of remaining cooldown ticks per hand slot.
            elixir: Current elixir level.
            card_defs: Card definition registry.
            arena: Current arena grid state.
            is_player_side: Whether deployment is on player's half.

        Returns:
            True if the action is valid.
        """
        if action.action_type == ActionType.PASS:
            return True

        if action.action_type == ActionType.INVALID:
            return False

        # Check card index is in hand
        if action.card_idx < 0 or action.card_idx >= len(hand):
            return False

        card_name = hand[action.card_idx]
        if card_name not in card_defs:
            return False

        card_def = card_defs[card_name]

        # Check cooldown
        if cooldowns[action.card_idx] > 0:
            return False

        # Check elixir cost
        if elixir < card_def.elixir_cost:
            return False

        # Check deployment zone validity
        if is_player_side:
            min_row = 3  # Player territory
        else:
            min_row = 0  # Opponent territory

        if action.target_row < min_row or action.target_row >= self.grid_rows:
            return False

        if action.target_col < 0 or action.target_col >= self.grid_cols:
            return False

        # Check cell isn't occupied (simplified)
        col_int = int(action.target_col)
        row_int = int(action.target_row)
        if arena[row_int, col_int] != 0:
            return False

        return True
