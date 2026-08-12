#!/usr/bin/env python3
"""Launch live-game agent for CR-Pipeline.

Usage:
    python scripts/play_live.py [--checkpoint runs/best/weights.pt] [--config configs/live_game.yaml]

Options:
    --checkpoint    Path to agent checkpoint to load
    --config        Live game config YAML file
    --overlay       Enable visualization overlay
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import EvolutionaryAgent, AgentConfig


def main():
    parser = argparse.ArgumentParser(
        description="Play Clash Royale with a trained AI agent."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to agent checkpoint")
    parser.add_argument("--config", type=str, default="configs/live_game.yaml",
                       help="Live game config YAML")
    parser.add_argument("--overlay", action="store_true",
                       help="Enable visualization overlay")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("play_live")

    logger.info("CR-Pipeline: Live Game Agent")
    logger.info("=" * 40)

    # Load agent
    agent = EvolutionaryAgent(
        config=AgentConfig(device="cpu"),
        seed=42,
    )

    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        agent.load_checkpoint(args.checkpoint)
        logger.info(f"Loaded agent: {agent}")
    else:
        logger.warning("No checkpoint specified. Using random agent.")

    # Import live game modules
    try:
        from src.env.live.screen_capture import ScreenCapture as LiveCapture
        from src.env.live.game_state import GameStateExtractor
        from src.env.live.action_mapper import ActionMapper
    except ImportError as e:
        logger.error(f"Failed to import live game modules: {e}")
        logger.error("Install required dependencies: pip install mss pyautogui opencv-python")
        return

    # Initialize live game components
    logger.info("Initializing live game components...")

    capture = LiveCapture()
    if not capture.start():
        logger.error("Failed to start screen capture.")
        return

    state_extractor = GameStateExtractor()
    action_mapper = ActionMapper()

    logger.info("Starting live gameplay...")
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            # Capture frame
            frame = capture.capture_frame()
            if frame is None:
                logger.warning("Failed to capture frame.")
                continue

            # Extract game state
            state = state_extractor.extract_state(frame)
            if state is None:
                continue

            # Preprocess state for neural net
            from src.env.sim import preprocess_state
            input_state = preprocess_state(state, resolution=64)

            # Select action
            action = agent.select_action(input_state)

            # Execute action
            action_mapper.execute_action(action)

    except KeyboardInterrupt:
        logger.info("Stopping live gameplay.")
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
