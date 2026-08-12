"""Live game screen capture module.

Provides screen capture and preprocessing for the Clash Royale game window.
Supports multiple capture backends (mss, pyautogui, dxcam).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """Configuration for screen capture.

    Attributes:
        target_window: Window title to capture.
        capture_method: Capture backend ("mss", "pyautogui", "dxcam").
        resolution: Capture resolution [width, height].
        target_resolution: Output resolution [width, height].
        frame_rate: Target FPS.
        color_format: Output color format ("rgb", "bgr", "gray").
    """
    target_window: str = "Clash Royale"
    capture_method: str = "mss"
    resolution: list = field(default_factory=lambda: [1280, 720])
    target_resolution: list = field(default_factory=lambda: [256, 256])
    frame_rate: int = 15
    color_format: str = "rgb"


class ScreenCapture:
    """Captures the Clash Royale game window in real-time.

    Captures frames at the target frame rate, resizes to the
    neural network input resolution, and converts to the
    specified color format.
    """

    def __init__(self, config: Optional[CaptureConfig] = None):
        """Initialize screen capture.

        Args:
            config: Capture configuration.
        """
        self.config = config or CaptureConfig()
        self.is_capturing = False
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.fps = 0.0

    def start(self) -> bool:
        """Start screen capture.

        Returns:
            True if capture started successfully.
        """
        try:
            method = self.config.capture_method
            if method == "mss":
                import mss
                self.sct = mss.mss()
            elif method == "pyautogui":
                import pyautogui
                self.pyautogui = pyautogui
            elif method == "dxcam":
                from dxcam import DXCam
                self.camera = DXCam()
            else:
                logger.error(f"Unknown capture method: {method}")
                return False

            self.is_capturing = True
            logger.info(f"Started screen capture: {method} @ "
                       f"{self.config.resolution[0]}x{self.config.resolution[1]}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for {method}: {e}")
            return False

    def stop(self) -> None:
        """Stop screen capture."""
        self.is_capturing = False
        logger.info("Stopped screen capture")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the game window.

        Returns:
            Captured frame as numpy array (H, W, C), or None.
        """
        if not self.is_capturing:
            return None

        import time
        now = time.time()

        if self.last_frame_time > 0:
            self.fps = 1.0 / (now - self.last_frame_time)

        self.last_frame_time = now
        self.frame_count += 1

        method = self.config.capture_method
        try:
            if method == "mss":
                monitor = {"left": 0, "top": 0,
                          "width": self.config.resolution[0],
                          "height": self.config.resolution[1]}
                screenshot = self.sct.grab(monitor)
                frame = np.array(screenshot)
                frame = frame[:, :, :3]  # BGRA -> RGB

            elif method == "pyautogui":
                frame = self.pyautogui.screenshot(
                    region=(0, 0,
                           self.config.resolution[0],
                           self.config.resolution[1])
                )
                frame = np.array(frame)
                frame = frame[:, :, ::-1]  # RGB -> BGR

            else:
                logger.warning(f"Capture method {method} not implemented")
                return None

            # Resize to target resolution
            th, tw = self.config.target_resolution
            if frame.shape[:2] != (th, tw):
                frame = self._resize(frame, th, tw)

            # Color format conversion
            if self.config.color_format == "gray":
                frame = np.mean(frame, axis=2, keepdims=True)
            elif self.config.color_format == "bgr":
                frame = frame[:, :, ::-1]

            return frame

        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None

    def _resize(self, frame: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Resize frame using nearest-neighbor interpolation."""
        h, w = frame.shape[:2]
        scale_x = w / tw
        scale_y = h / th

        resized = np.zeros((th, tw, frame.shape[2]), dtype=frame.dtype)
        for y in range(th):
            for x in range(tw):
                src_y = min(int(y * scale_y), h - 1)
                src_x = min(int(x * scale_x), w - 1)
                resized[y, x] = frame[src_y, src_x]

        return resized

    def get_fps(self) -> float:
        """Get current capture FPS."""
        return self.fps
