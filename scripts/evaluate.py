#!/usr/bin/env python3
"""Evaluate trained agents in CR-Pipeline.

Usage:
    python scripts/evaluate.py [--checkpoint runs/best/weights.pt] [--matches 20] [--opponent random|greedy|elite] [--tournament]

Options:
    --checkpoint    Path to agent weights to evaluate
    --matches       Number of matches to play
    --opponent      Opponent type: random, greedy, elite
    --tournament    Run tournament mode against population
    --population    Path to population checkpoint for tournament
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.train import FitnessEvaluator
from src.env.sim import SimulationEngine


def load_checkpoint(path: str) -> np.ndarray:
    """Load agent weights from a checkpoint file.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Numpy array of weights.
    """
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        return np.array(checkpoint["weights"])
    return np.array(checkpoint)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Clash Royale agents."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to agent weights checkpoint")
    parser.add_argument("--population", type=str, default=None,
                       help="Path to population checkpoint for tournament")
    parser.add_argument("--matches", type=int, default=20,
                       help="Number of matches per agent")
    parser.add_argument("--opponent", type=str, default="random",
                       choices=["random", "greedy", "elite"],
                       help="Opponent type")
    parser.add_argument("--tournament", action="store_true",
                       help="Run tournament mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("evaluate")

    evaluator = FitnessEvaluator(num_workers=4, matches_per_agent=args.matches)

    if args.tournament:
        # Tournament mode
        if not args.population:
            logger.error("--population is required for tournament mode")
            return

        # Load population
        checkpoint = torch.load(args.population, map_location="cpu")
        weights_list = [np.array(w) for w in checkpoint["weights"]]

        logger.info(f"Running tournament with {len(weights_list)} agents...")
        result = evaluator.run_tournament(
            weights_list=weights_list,
            matches_per_agent=args.matches,
            seed=args.seed,
        )

        logger.info("Tournament Results:")
        logger.info("-" * 40)
        for rank, (agent_id, score) in enumerate(result.rankings, 1):
            wins = result.wins.get(agent_id, 0)
            draws = result.draws.get(agent_id, 0)
            losses = result.losses.get(agent_id, 0)
            logger.info(f"  #{rank} {agent_id}: score={score:.2f} "
                       f"(W={wins}, D={draws}, L={losses})")

    elif args.checkpoint:
        # Single agent evaluation
        logger.info(f"Evaluating agent from {args.checkpoint}")
        weights = load_checkpoint(args.checkpoint)

        # Evaluate against different opponent types
        for opp_type in ["random", "greedy"]:
            result = evaluator.evaluate_single(
                weights=weights,
                matches=args.matches,
                opponent_type=opp_type,
                seed=args.seed,
            )
            logger.info(f"vs {opp_type}: fitness={result.fitness:.3f} "
                       f"(W={result.wins}, D={result.draws}, L={result.losses})")

    else:
        logger.error("Specify --checkpoint or --tournament")
        return

    evaluator.shutdown()
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
