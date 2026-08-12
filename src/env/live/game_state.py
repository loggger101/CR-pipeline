"""Game state extraction from captured frames.

Provides:
- Template matching for card identification
- Color detection for elixir levels
- Tower health estimation
- Unit position detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExtractedState:
    """Extracted game state from a captured frame.

    Attributes:
        card_names: Names of cards in the current hand.
        card_cooldowns: Remaining cooldown for each card slot.
        player_elixir: Current player elixir level (0-10).
        opponent_elixir: Current opponent elixir level (0-10).
        tower_health: Dict of tower name -> HP percentage (0-1).
        unit_positions: List of (col, row, type, owner) tuples.
        time_remaining: Seconds remaining in the match.
        is_overtime: Whether the match is in overtime.
        arena_frame: Processed arena frame for visualization.
    """
    card_names: List[str] = field(default_factory=list)
    card_cooldowns: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    player_elixir: float = 5.0
    opponent_elixir: float = 5.0
    tower_health: Dict[str, float] = field(default_factory=dict)
    unit_positions: List[Tuple[float, float, str, str]] = field(default_factory=list)
    time_remaining: float = 180.0
    is_overtime: bool = False
    arena_frame: Optional[np.ndarray] = None


class GameStateExtractor:
    """Extracts game state from captured frames.

    Uses template matching, color detection, and CNN classifiers
    to identify cards, elixir levels, tower health, and unit positions.
    """

    def __init__(self, method: str = "template_matching"):
        """Initialize the state extractor.

        Args:
            method: Extraction method ("template_matching", "color_detection", "cnn_classifier").
        """
        self.method = method
        self.template_cache: Dict[str, np.ndarray] = {}

    def extract_state(self, frame: np.ndarray) -> Optional[ExtractedState]:
        """Extract game state from a captured frame.

        Args:
            frame: Game frame (H, W, 3) in RGB or BGR.

        Returns:
            ExtractedState, or None if extraction failed.
        """
        state = ExtractedState()

        if self.method == "template_matching":
            state = self._extract_template_matching(frame)
        elif self.method == "color_detection":
            state = self._extract_color_detection(frame)
        elif self.method == "cnn_classifier":
            state = self._extract_cnn(frame)
        else:
            logger.error(f"Unknown extraction method: {self.method}")
            return None

        return state

    def _extract_template_matching(self, frame: np.ndarray) -> ExtractedState:
        """Extract state using template matching.

        Uses pre-saved card templates to identify cards in the hand.
        """
        state = ExtractedState()

        # Card regions (from config)
        card_regions = [
            {"x": 100, "y": 550, "w": 80, "h": 110},
            {"x": 200, "y": 550, "w": 80, "h": 110},
            {"x": 300, "y": 550, "w": 80, "h": 110},
            {"x": 400, "y": 550, "w": 80, "h": 110},
        ]

        # Extract card images and match against templates
        for i, region in enumerate(card_regions):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                card_roi = frame[y:y+h, x:x+w]
                # Template matching would go here
                # For now, return placeholder
                state.card_names.append(f"card_{i}")

        # Elixir detection
        state.player_elixir = self._detect_elixir(frame, "player")
        state.opponent_elixir = self._detect_elixir(frame, "opponent")

        # Tower health
        state.tower_health = self._detect_tower_health(frame)

        return state

    def _extract_color_detection(self, frame: np.ndarray) -> ExtractedState:
        """Extract state using color detection.

        Uses color ranges to identify game elements.
        """
        state = ExtractedState()

        # Convert to HSV for color detection
        if len(frame.shape) == 3:
            hsv = self._bgr_to_hsv(frame)
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

        # Elixir (purple color)
        lower_purple = np.array([120, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)
        state.player_elixir = float(np.sum(mask > 0)) / mask.size * 10

        # Tower health (red for damaged towers)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        state.tower_health["detected"] = float(np.sum(mask_red > 0)) / mask_red.size

        return state

    def _extract_cnn(self, frame: np.ndarray) -> ExtractedState:
        """Extract state using a CNN classifier.

        Requires a trained model file.
        """
        state = ExtractedState()

        # Placeholder: would run inference on a trained model
        # state.card_names = model.predict_cards(frame)
        # state.tower_health = model.predict_towers(frame)

        return state

    def _detect_elixir(self, frame: np.ndarray, player: str) -> float:
        """Detect elixir level from the game frame.

        Args:
            frame: Game frame.
            player: "player" or "opponent".

        Returns:
            Elixir level (0-10).
        """
        # Elixir bar region (from config)
        elixir_bar = {"x": 1060, "y": 640, "w": 120, "h": 40}

        if player == "player":
            x, y, w, h = 50, 640, 120, 40  # Player elixir position
        else:
            x, y, w, h = 1060, 640, 120, 40  # Opponent elixir position

        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return 5.0

        elixir_roi = frame[y:y+h, x:x+w]
        # Count purple pixels
        if len(elixir_roi.shape) == 3:
            hsv = self._bgr_to_hsv(elixir_roi)
            lower_purple = np.array([120, 50, 50])
            upper_purple = np.array([160, 255, 255])
            mask = cv2.inRange(hsv, lower_purple, upper_purple)
            fill_ratio = np.sum(mask > 0) / mask.size
            return fill_ratio * 10.0

        return 5.0  # Default

    def _detect_tower_health(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect tower health from the game frame.

        Args:
            frame: Game frame.

        Returns:
            Dict of tower name -> HP percentage.
        """
        # Tower health regions (from config)
        tower_regions = [
            ("opp_princess_left", 200, 100, 60, 30),
            ("opp_princess_right", 1000, 100, 60, 30),
            ("opp_king", 600, 50, 60, 30),
            ("player_princess_left", 200, 600, 60, 30),
            ("player_princess_right", 1000, 600, 60, 30),
        ]

        health = {}
        for name, x, y, w, h in tower_regions:
            if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                roi = frame[y:y+h, x:x+w]
                # Detect color to estimate health
                if len(roi.shape) == 3:
                    hsv = self._bgr_to_hsv(roi)
                    # Green = full health, red = damaged
                    lower_green = np.array([40, 50, 50])
                    upper_green = np.array([80, 255, 255])
                    mask = cv2.inRange(hsv, lower_green, upper_green)
                    health[name] = float(np.sum(mask > 0)) / mask.size
                else:
                    health[name] = 1.0
            else:
                health[name] = 0.0

        return health

    def _bgr_to_hsv(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to HSV color space."""
        try:
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except ImportError:
            # Fallback: simple conversion
            hsv = np.zeros_like(frame, dtype=np.float32)
            rgb = frame.astype(np.float32) / 255.0
            cmax = rgb.max(axis=2, keepdims=True)
            cmin = rgb.min(axis=2, keepdims=True)
            diff = cmax - cmin
            hsv[:, :, 2] = cmax[:, :, 0]  # Value
            if cmax[:, :, 0].any():
                hsv[:, :, 1] = diff[:, :, 0] / cmax[:, :, 0]  # Saturation
            return hsv

    def register_template(self, name: str, template: np.ndarray) -> None:
        """Register a card template for matching.

        Args:
            name: Card name.
            template: Template image array.
        """
        self.template_cache[name] = template

    def get_matching_score(self, frame_roi: np.ndarray,
                           template_name: str) -> float:
        """Get template matching score for a region.

        Args:
            frame_roi: Region of interest from the frame.
            template_name: Name of the registered template.

        Returns:
            Matching score (0-1).
        """
        if template_name not in self.template_cache:
            return 0.0

        template = self.template_cache[template_name]
        # Normalized cross-correlation
        try:
            import cv2
            result = cv2.matchTemplate(frame_roi, template, cv2.TM_CCOEFF_NORMED)
            return float(cv2.minMaxLoc(result)[1])
        except ImportError:
            # Fallback: simple correlation
            template = cv2.resize(template, frame_roi.shape[:2][::-1])
            corr = np.corrcoef(frame_roi.flatten(), template.flatten())[0, 1]
            return max(0, corr)
