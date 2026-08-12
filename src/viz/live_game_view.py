"""Live game visualization overlay.

Provides:
- Real-time screen capture integration
- Neural network prediction overlay
- Fitness tracking display
- Debug visualization for live gameplay
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LiveOverlayData:
    """Data for the live game overlay.

    Attributes:
        fps: Current capture FPS.
        card_hand: Current card hand (list of card names).
        player_elixir: Current player elixir level.
        opponent_elixir: Current opponent elixir level.
        predicted_action: Predicted action from the neural net.
        confidence: Confidence of the prediction.
        tower_health: Tower health percentages.
        unit_density: Simplified unit density map.
        fitness_score: Current fitness score.
        generation: Current training generation.
        is_alive: Whether the overlay is active.
    """
    fps: float = 0.0
    card_hand: List[str] = field(default_factory=list)
    player_elixir: float = 5.0
    opponent_elixir: float = 5.0
    predicted_action: Optional[dict] = None
    confidence: float = 0.0
    tower_health: Dict[str, float] = field(default_factory=dict)
    unit_density: np.ndarray = field(default_factory=lambda: np.zeros((6, 8)))
    fitness_score: float = 0.0
    generation: int = 0
    is_alive: bool = False


class LiveGameView:
    """Manages the live game visualization overlay.

    Integrates with the screen capture module to display:
    - Card hand with cooldown indicators
    - Elixir levels for both players
    - Predicted actions (card selection + placement preview)
    - Tower health bars
    - Unit density heatmap
    - FPS counter
    - Fitness score display
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize the live game view.

        Args:
            config: Overlay configuration.
        """
        self.config = config or {}
        self.overlay_data = LiveOverlayData()
        self.frame_buffer: List[np.ndarray] = []
        self.fps_history: List[float] = []
        self.last_frame_time: float = 0.0

        # Overlay settings
        self.enabled = self.config.get("overlay", {}).get("enabled", True)
        self.color_scheme = self.config.get("overlay", {}).get(
            "color_scheme", "default"
        )

        # Color schemes
        self.colors = {
            "default": {
                "background": (30, 30, 40),
                "text": (255, 255, 255),
                "card": (100, 150, 255),
                "card_selected": (255, 200, 50),
                "elixir": (150, 100, 255),
                "tower_player": (50, 200, 50),
                "tower_opponent": (200, 50, 50),
                "unit_player": (50, 180, 50),
                "unit_opponent": (180, 50, 50),
                "placement": (255, 255, 100),
                "fps": (100, 255, 100),
            },
            "high_contrast": {
                "background": (0, 0, 0),
                "text": (255, 255, 255),
                "card": (0, 255, 255),
                "card_selected": (255, 255, 0),
                "elixir": (255, 0, 255),
                "tower_player": (0, 255, 0),
                "tower_opponent": (255, 0, 0),
                "unit_player": (0, 255, 0),
                "unit_opponent": (255, 0, 0),
                "placement": (255, 255, 0),
                "fps": (0, 255, 0),
            },
            "minimal": {
                "background": (0, 0, 0),
                "text": (200, 200, 200),
                "card": (150, 150, 150),
                "card_selected": (255, 255, 255),
                "elixir": (150, 150, 150),
                "tower_player": (100, 100, 100),
                "tower_opponent": (100, 100, 100),
                "unit_player": (100, 100, 100),
                "unit_opponent": (100, 100, 100),
                "placement": (200, 200, 100),
                "fps": (100, 100, 100),
            },
        }

    def update(self, overlay_data: LiveOverlayData) -> None:
        """Update the overlay data.

        Args:
            overlay_data: New overlay data to display.
        """
        self.overlay_data = overlay_data
        self.overlay_data.is_alive = True

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render the overlay onto a game frame.

        Args:
            frame: Game frame (H, W, 3) in BGR or RGB.

        Returns:
            Frame with overlay rendered.
        """
        if not self.enabled:
            return frame

        colors = self.colors.get(self.color_scheme, self.colors["default"])
        overlay = frame.copy().astype(np.float32)

        # Draw FPS
        if self.config.get("overlay", {}).get("fps_display", True):
            fps_text = f"FPS: {self.overlay_data.fps:.1f}"
            # Simple text rendering placeholder
            # In practice, would use cv2.putText or similar

        # Draw card hand
        if self.config.get("overlay", {}).get("draw_card_selection", True):
            self._draw_card_hand(overlay, colors)

        # Draw action preview
        if self.config.get("overlay", {}).get("draw_action_preview", True):
            self._draw_action_preview(overlay, colors)

        # Draw tower health
        if self.config.get("overlay", {}).get("draw_tower_health", True):
            self._draw_tower_health(overlay, colors)

        # Draw fitness
        if self.config.get("overlay", {}).get("fitness_display", True):
            self._draw_fitness(overlay, colors)

        return overlay.astype(np.uint8)

    def _draw_card_hand(self, frame: np.ndarray, colors: dict) -> None:
        """Draw the card hand overlay."""
        if not self.overlay_data.card_hand:
            return

        # Placeholder: would draw card rectangles with names
        # and cooldown indicators in the bottom portion of the screen

    def _draw_action_preview(self, frame: np.ndarray, colors: dict) -> None:
        """Draw the predicted action preview."""
        if self.overlay_data.predicted_action is None:
            return

        # Draw placement preview on the arena
        action = self.overlay_data.predicted_action
        if "target_col" in action and "target_row" in action:
            col = int(action["target_col"])
            row = int(action["target_row"])
            # Highlight the target cell
            if 0 <= row < frame.shape[0] and 0 <= col < frame.shape[1]:
                # Draw a semi-transparent overlay on target cell
                pass

    def _draw_tower_health(self, frame: np.ndarray, colors: dict) -> None:
        """Draw tower health bars."""
        for tower_name, health_pct in self.overlay_data.tower_health.items():
            # Draw health bar next to tower position
            # Placeholder for actual rendering
            pass

    def _draw_fitness(self, frame: np.ndarray, colors: dict) -> None:
        """Draw fitness score display."""
        if self.overlay_data.fitness_score != 0:
            # Draw fitness score in corner
            text = f"Fitness: {self.overlay_data.fitness_score:.2f}"
            # Placeholder for actual text rendering
            pass

    def get_fps(self) -> float:
        """Get the current FPS from the frame buffer."""
        if len(self.fps_history) > 0:
            return float(np.mean(self.fps_history[-60:]))
        return self.overlay_data.fps

    def record_frame(self, timestamp: float) -> None:
        """Record a frame timestamp for FPS calculation."""
        if self.last_frame_time > 0:
            fps = 1.0 / (timestamp - self.last_frame_time)
            self.fps_history.append(fps)
            # Keep last 60 entries
            if len(self.fps_history) > 60:
                self.fps_history = self.fps_history[-60:]
        self.last_frame_time = timestamp

    def reset(self) -> None:
        """Reset the overlay state."""
        self.overlay_data.is_alive = False
        self.overlay_data.fitness_score = 0.0
        self.overlay_data.predicted_action = None
        self.fps_history = []
        self.last_frame_time = 0.0


class ScreenCapture:
    """Screen capture module for live gameplay.

    Captures the Clash Royale game window and provides
    preprocessed frames for the neural network.
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize screen capture.

        Args:
            config: Screen capture configuration.
        """
        self.config = config or {}
        self.target_window = self.config.get("screen_capture", {}).get(
            "target_window", "Clash Royale"
        )
        self.capture_method = self.config.get("screen_capture", {}).get(
            "capture_method", "mss"
        )
        self.resolution = tuple(
            self.config.get("screen_capture", {}).get("resolution", [1280, 720])
        )
        self.target_resolution = tuple(
            self.config.get("screen_capture", {}).get("target_resolution", [256, 256])
        )
        self.frame_rate = self.config.get("screen_capture", {}).get("frame_rate", 15)

        self.is_capturing = False
        self.frame_count = 0
        self.last_capture_time = 0.0

    def start(self) -> bool:
        """Start screen capture.

        Returns:
            True if capture started successfully.
        """
        try:
            if self.capture_method == "mss":
                import mss
                self.sct = mss.mss()
            elif self.capture_method == "pyautogui":
                import pyautogui
                self.pyautogui = pyautogui
            elif self.capture_method == "dxcam":
                from dxcam import DXCam
                self.camera = DXCam()
            else:
                logger.error(f"Unknown capture method: {self.capture_method}")
                return False

            self.is_capturing = True
            logger.info(f"Started screen capture: {self.capture_method}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for {self.capture_method}: {e}")
            return False

    def stop(self) -> None:
        """Stop screen capture."""
        self.is_capturing = False
        logger.info("Stopped screen capture")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the game window.

        Returns:
            Captured frame as numpy array, or None if capture failed.
        """
        if not self.is_capturing:
            return None

        # Capture timestamp for FPS
        import time
        now = time.time()
        self.frame_count += 1

        # Capture frame (method-specific)
        if self.capture_method == "mss":
            try:
                monitor = {"left": 0, "top": 0, "width": self.resolution[0],
                          "height": self.resolution[1]}
                screenshot = self.sct.grab(monitor)
                frame = np.array(screenshot)
                # Convert BGRA to RGB
                frame = frame[:, :, :3]
            except Exception as e:
                logger.error(f"Capture error: {e}")
                return None

        elif self.capture_method == "pyautogui":
            try:
                frame = self.pyautogui.screenshot(
                    region=(0, 0, self.resolution[0], self.resolution[1])
                )
                frame = np.array(frame)
                frame = frame[:, :, ::-1]  # RGB to BGR
            except Exception as e:
                logger.error(f"Capture error: {e}")
                return None

        else:
            logger.warning(f"Capture method {self.capture_method} not implemented")
            return None

        # Resize to target resolution
        if frame.shape[:2] != self.target_resolution:
            frame = self._resize_frame(frame)

        self.last_capture_time = now
        return frame

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target resolution using nearest neighbor."""
        h, w = frame.shape[:2]
        th, tw = self.target_resolution

        # Simple nearest-neighbor resize
        scale_x = w / tw
        scale_y = h / th

        resized = np.zeros((th, tw, frame.shape[2]), dtype=frame.dtype)
        for y in range(th):
            for x in range(tw):
                src_y = min(int(y * scale_y), h - 1)
                src_x = min(int(x * scale_x), w - 1)
                resized[y, x] = frame[src_y, src_x]

        return resized
