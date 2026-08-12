"""Action mapper for live gameplay.

Maps neural network outputs to in-game actions via:
- Mouse click simulation
- Keyboard hotkey simulation
- Hybrid approach (hotkeys for card selection, mouse for placement)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionConfig:
    """Configuration for action mapping.

    Attributes:
        method: Action method ("mouse_click", "hotkey", "hybrid").
        grid_resolution: Grid resolution [cols, rows].
        deployment_zone: Deployment zone ("player_half", "full").
        debounce_ms: Minimum time between actions in ms.
        cooldown_validation: Whether to validate card cooldowns.
        hotkey_map: Mapping of action names to key names.
    """
    method: str = "mouse_click"
    grid_resolution: list = field(default_factory=lambda: [8, 6])
    deployment_zone: str = "player_half"
    debounce_ms: int = 100
    cooldown_validation: bool = True
    hotkey_map: Dict[str, str] = field(default_factory=lambda: {
        "select_card_1": "1",
        "select_card_2": "2",
        "select_card_3": "3",
        "select_card_4": "4",
        "confirm_placement": "mouse_left",
    })


class ActionMapper:
    """Maps neural network actions to in-game inputs.

    Handles:
    - Debouncing to prevent rapid actions
    - Cooldown validation
    - Grid-to-screen coordinate conversion
    - Mouse and keyboard simulation
    """

    def __init__(self, config: Optional[ActionConfig] = None):
        """Initialize the action mapper.

        Args:
            config: Action mapping configuration.
        """
        self.config = config or ActionConfig()
        self.last_action_time = 0.0
        self.card_cooldowns: Dict[int, float] = {}
        self.selected_card: Optional[int] = None

        # Initialize input simulation
        self._init_input_sim()

    def _init_input_sim(self) -> None:
        """Initialize input simulation libraries."""
        try:
            import pyautogui
            self.pyautogui = pyautogui
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False
            logger.info("Initialized pyautogui for input simulation")
        except ImportError:
            logger.error("pyautogui not installed. Install with: pip install pyautogui")

    def execute_action(self, action: dict) -> bool:
        """Execute a neural network action as an in-game input.

        Args:
            action: Action dict with keys:
                - card_idx: Card index (0-3) or -1 for pass
                - target_col: Target column
                - target_row: Target row
                - is_exploration: Whether this is an exploratory action

        Returns:
            True if action was executed successfully.
        """
        card_idx = action.get("card_idx", -1)
        target_col = action.get("target_col", -1)
        target_row = action.get("target_row", -1)

        # Check debounce
        now = time.time()
        if (now - self.last_action_time) * 1000 < self.config.debounce_ms:
            return False

        # Validate cooldown
        if self.config.cooldown_validation:
            if card_idx in self.card_cooldowns:
                if self.card_cooldowns[card_idx] > 0:
                    return False

        # Execute based on method
        if card_idx == -1:
            # Pass action
            self._execute_pass()
        elif self.config.method == "mouse_click":
            self._execute_mouse_click(card_idx, target_col, target_row)
        elif self.config.method == "hotkey":
            self._execute_hotkey(card_idx, target_col, target_row)
        elif self.config.method == "hybrid":
            self._execute_hybrid(card_idx, target_col, target_row)

        self.last_action_time = now
        return True

    def _execute_mouse_click(self, card_idx: int,
                             target_col: float, target_row: float) -> None:
        """Execute action via mouse clicks.

        1. Click on card in hand
        2. Move to target position
        3. Click to deploy
        """
        try:
            import pyautogui

            # Get card position
            card_positions = [
                (100, 550), (200, 550), (300, 550), (400, 550)
            ]
            if card_idx < len(card_positions):
                pyautogui.click(card_positions[card_idx])

            # Get target position
            grid_w, grid_h = self.config.grid_resolution
            target_x = (target_col + 0.5) / grid_w * 1280  # Map to screen coords
            target_y = (target_row + 0.5) / grid_h * 720
            pyautogui.click(target_x, target_y)

            self.selected_card = card_idx

        except Exception as e:
            logger.error(f"Mouse click action failed: {e}")

    def _execute_hotkey(self, card_idx: int,
                        target_col: float, target_row: float) -> None:
        """Execute action via keyboard hotkeys.

        1. Press hotkey to select card
        2. Move mouse to target position
        3. Press confirm key
        """
        try:
            import pyautogui

            # Select card via hotkey
            key_map = self.config.hotkey_map
            select_key = key_map.get(f"select_card_{card_idx + 1}", str(card_idx + 1))
            pyautogui.press(select_key)

            # Move to target
            grid_w, grid_h = self.config.grid_resolution
            target_x = (target_col + 0.5) / grid_w * 1280
            target_y = (target_row + 0.5) / grid_h * 720
            pyautogui.moveTo(target_x, target_y)

            # Confirm
            confirm = key_map.get("confirm_placement", "mouse_left")
            if confirm == "mouse_left":
                pyautogui.click()

        except Exception as e:
            logger.error(f"Hotkey action failed: {e}")

    def _execute_hybrid(self, card_idx: int,
                        target_col: float, target_row: float) -> None:
        """Execute action via hybrid method.

        Uses hotkeys for card selection and mouse for placement.
        """
        self._execute_hotkey(card_idx, target_col, target_row)

    def _execute_pass(self) -> None:
        """Execute a pass action (no action this tick)."""
        pass

    def update_cooldowns(self, cooldowns: List[int]) -> None:
        """Update card cooldown state.

        Args:
            cooldowns: List of remaining cooldown ticks per card slot.
        """
        self.card_cooldowns = {i: cd for i, cd in enumerate(cooldowns)}

    def get_card_positions(self) -> List[Tuple[int, int]]:
        """Get the screen positions of card slots.

        Returns:
            List of (x, y) positions for each card slot.
        """
        return [
            (100, 550), (200, 550), (300, 550), (400, 550)
        ]

    def grid_to_screen(self, col: float, row: float) -> Tuple[int, int]:
        """Convert grid coordinates to screen coordinates.

        Args:
            col: Grid column.
            row: Grid row.

        Returns:
            Screen (x, y) coordinates.
        """
        grid_w, grid_h = self.config.grid_resolution
        x = int((col + 0.5) / grid_w * 1280)
        y = int((row + 0.5) / grid_h * 720)
        return x, y

    def screen_to_grid(self, x: int, y: int) -> Tuple[float, float]:
        """Convert screen coordinates to grid coordinates.

        Args:
            x: Screen x coordinate.
            y: Screen y coordinate.

        Returns:
            Grid (col, row) coordinates.
        """
        grid_w, grid_h = self.config.grid_resolution
        col = (x / 1280) * grid_w - 0.5
        row = (y / 720) * grid_h - 0.5
        return col, row
