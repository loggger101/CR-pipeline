"""Live game visualization for real-time Clash Royale gameplay.

Provides:
- Real-time overlay rendering on the game window
- Screen capture integration
- Neural net prediction visualization
- Fitness tracking overlay
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rendering import SimulationRenderer

logger = logging.getLogger(__name__)


@dataclass
class LiveGameOverlay:
    """Overlay data for live game visualization.

    Attributes:
        card_selection: Index of selected card (0-3) or -1.
        placement_target: (col, row) for placement preview.
        predicted_probs: Card selection probabilities.
        current_fitness: Current fitness score.
        elixir_level: Current elixir level.
        tower_health: Dict of tower name -> HP ratio.
        unit_positions: List of (col, row, owner, hp_ratio) tuples.
        fps: Current frame rate.
        latency_ms: Current processing latency.
    """
    card_selection: int = -1
    placement_target: Optional[Tuple[float, float]] = None
    predicted_probs: List[float] = field(default_factory=list)
    current_fitness: float = 0.0
    elixir_level: float = 5.0
    tower_health: Dict[str, float] = field(default_factory=dict)
    unit_positions: List[Tuple[float, float, str, float]] = field(default_factory=list)
    fps: float = 0.0
    latency_ms: float = 0.0


class LiveGameView:
    """Manages live game visualization overlay.

    Handles:
    - Screen capture integration
    - Overlay rendering
    - Neural net prediction display
    - Fitness tracking
    """

    def __init__(
        self,
        target_window: str = "Clash Royale",
        capture_resolution: Tuple[int, int] = (1280, 720),
        overlay_enabled: bool = True,
    ):
        self.target_window = target_window
        self.capture_resolution = capture_resolution
        self.overlay_enabled = overlay_enabled
        self.renderer = SimulationRenderer()

        # State tracking
        self.current_overlay: Optional[LiveGameOverlay] = None
        self.frame_buffer: List[np.ndarray] = []
        self.fps_history: List[float] = []
        self._last_frame_time: float = 0.0

        # Performance monitoring
        self.total_frames = 0
        self.total_processing_time = 0.0

    def update_overlay(self, overlay_data: LiveGameOverlay) -> None:
        """Update the overlay data.

        Args:
            overlay_data: New overlay information.
        """
        self.current_overlay = overlay_data
        self.total_frames += 1

    def render_frame(self, game_frame: np.ndarray) -> np.ndarray:
        """Render a frame with overlay on top of the game frame.

        Args:
            game_frame: Raw game frame (H, W, 3) BGR or RGB.

        Returns:
            Rendered frame with overlay.
        """
        if not self.overlay_enabled or self.current_overlay is None:
            return game_frame

        # Calculate FPS
        now = time.time()
        if self._last_frame_time > 0:
            fps = 1.0 / (now - self._last_frame_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 60:
                self.fps_history = self.fps_history[-60:]
        self._last_frame_time = now

        # Get current overlay
        overlay = self.current_overlay

        # Create overlay image
        overlay_img = self._create_overlay_image(overlay)

        # Composite overlay onto game frame
        h, w = game_frame.shape[:2]
        oh, ow = overlay_img.shape[:2]

        # Place overlay in bottom-right corner
        result = game_frame.copy()
        y_start = max(0, h - oh)
        x_start = max(0, w - ow)
        result[y_start:y_start + min(oh, h - y_start),
               x_start:x_start + min(ow, w - x_start)] = \
            overlay_img[:min(oh, h - y_start), :min(ow, w - x_start)]

        return result

    def _create_overlay_image(self, overlay: LiveGameOverlay) -> np.ndarray:
        """Create an overlay image from overlay data."""
        panel_height = 180
        panel_width = 280
        img = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        img[:, :] = (20, 20, 30)

        self._draw_card_selection(img, overlay)
        self._draw_elixir_bar(img, overlay.elixir_level, 40)
        self._draw_tower_health(img, overlay.tower_health, 80)
        self._draw_performance_stats(img, overlay.fps, overlay.latency_ms, 150)

        return img

    def _draw_card_selection(self, img, overlay):
        """Draw card selection info."""
        slot_width, slot_height = 45, 55
        start_x, y = 10, 10
        for i in range(4):
            x = start_x + i * (slot_width + 5)
            color = (255, 255, 0) if (i == overlay.card_selection) else (80, 80, 80)
            for dy in range(slot_height):
                for dx in range(slot_width):
                    if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                        img[y + dy, x + dx] = color
        if overlay.card_selection >= 0:
            sel_x = start_x + overlay.card_selection * (slot_width + 5)
            for dy in range(slot_height):
                for dx in range(slot_width):
                    if 0 <= y + dy < img.shape[0] and 0 <= sel_x + dx < img.shape[1]:
                        img[y + dy, sel_x + dx] = (255, 200, 0)

    def _draw_elixir_bar(self, img, elixir, y):
        """Draw elixir level bar."""
        bar_width, bar_height, x = 180, 10, 10
        ratio = elixir / 10.0
        for dy in range(bar_height):
            for dx in range(bar_width):
                if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                    img[y + dy, x + dx] = (50, 50, 50)
        fill_width = int(bar_width * ratio)
        for dy in range(bar_height):
            for dx in range(fill_width):
                if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                    img[y + dy, x + dx] = (180, 50, 200)
        text = f"Elixir: {elixir:.1f}"
        for i, c in enumerate(text):
            tx = x + i * 7
            if 0 <= y + 15 < img.shape[0] and 0 <= tx < img.shape[1]:
                img[y + 15, tx] = (255, 255, 255)

    def _draw_tower_health(self, img, tower_health, y):
        """Draw tower health bars."""
        for i, (name, ratio) in enumerate(tower_health.items()):
            bar_width, bar_height, x = 55, 6, 10 + i * 65
            color = (0, 255, 0) if ratio > 0.5 else (255, 0, 0) if ratio < 0.2 else (255, 255, 0)
            for dy in range(bar_height):
                for dx in range(bar_width):
                    if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                        img[y + dy, x + dx] = (50, 50, 50)
            fill_width = int(bar_width * ratio)
            for dy in range(bar_height):
                for dx in range(fill_width):
                    if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                        img[y + dy, x + dx] = color

    def _draw_performance_stats(self, img, fps, latency_ms, y):
        """Draw FPS and latency stats."""
        fps_text = f"FPS: {fps:.1f}"
        lat_text = f"Latency: {latency_ms:.0f}ms"
        for i, c in enumerate(fps_text):
            tx = 10 + i * 7
            if 0 <= y < img.shape[0] and 0 <= tx < img.shape[1]:
                img[y, tx] = (100, 255, 100)
        for i, c in enumerate(lat_text):
            tx = 10 + i * 7
            if 0 <= y + 20 < img.shape[0] and 0 <= tx < img.shape[1]:
                img[y + 20, tx] = (100, 100, 255)

    def get_performance_stats(self):
        """Get current performance statistics."""
        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        return {
            "fps": avg_fps,
            "total_frames": self.total_frames,
            "avg_latency_ms": (self.total_processing_time / self.total_frames * 1000)
                if self.total_frames > 0 else 0,
        }

    def reset(self):
        """Reset state tracking."""
        self.current_overlay = None
        self.frame_buffer = []
        self.fps_history = []
        self._last_frame_time = 0.0
        self.total_frames = 0
        self.total_processing_time = 0.0
