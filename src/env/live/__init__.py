"""CR-Pipeline: Live Game Environment.

Provides:
- Screen capture for the Clash Royale game window
- Game state extraction from captured frames
- Action mapping from neural network outputs to in-game inputs
"""

from .screen_capture import ScreenCapture, CaptureConfig
from .game_state import GameStateExtractor, ExtractedState
from .action_mapper import ActionMapper, ActionConfig

__all__ = [
    "ScreenCapture",
    "CaptureConfig",
    "GameStateExtractor",
    "ExtractedState",
    "ActionMapper",
    "ActionConfig",
]
