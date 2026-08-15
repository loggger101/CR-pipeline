#!/usr/bin/env python3
"""Evaluate trained agents in CR-Pipeline.

Usage:
    python scripts/evaluate.py --checkpoint runs/<run>/best/best_agent.pt [--matches 20] [--opponent all] [--tournament]

Options:
    --checkpoint    Path to agent weights to evaluate
    --matches       Number of matches to play
    --opponent      random, greedy, balanced, aggressive, defensive, or all
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
from src.env.sim.parallel_runner import _OPPONENT_ACTIONS
from src.models.policy import DEFAULT_POLICY_SPEC
from src.serialization import load_checkpoint as _load_ckpt_file

SCRIPTED_OPPONENTS = sorted(_OPPONENT_ACTIONS)


def load_checkpoint(path: str) -> np.ndarray:
    """Load an agent's policy genome from a checkpoint file.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Numpy array holding the policy genome.

    Raises:
        ValueError: if the checkpoint holds Torch network parameters rather
            than a policy genome, which cannot be played by the simulator.
    """
    # weights_only=False: these checkpoints carry numpy arrays and metadata,
    # not just tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return np.asarray(checkpoint)

    for key in ("genome", "weights"):
        if checkpoint.get(key) is not None:
            weights = np.asarray(checkpoint[key])
            break
    else:
        raise ValueError(f"{path} contains no agent parameters")

    expected = DEFAULT_POLICY_SPEC.num_params
    if weights.size != expected:
        raise ValueError(
            f"{path} holds {weights.size} parameters but the simulator plays "
            f"{expected}-parameter policy genomes. This checkpoint most likely "
            f"stores Torch network weights, which the match runner cannot use."
        )
    return weights


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
    parser.add_argument("--opponent", type=str, default="all",
                       choices=SCRIPTED_OPPONENTS + ["all"],
                       help="Opponent to play, or 'all' for every baseline")
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

        # Load population. This is the raw file loader, not this module's
        # load_checkpoint, which returns a single agent's genome.
        checkpoint = _load_ckpt_file(args.population)
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

        # Honour --opponent. This previously hardcoded random/greedy and
        # ignored the flag entirely.
        opponents = (SCRIPTED_OPPONENTS if args.opponent == "all"
                     else [args.opponent])
        for opp_type in opponents:
            result = evaluator.evaluate_against_opponent(
                weights=weights,
                matches=args.matches,
                opponent_type=opp_type,
                seed=args.seed,
            )
            played = max(1, result.wins + result.draws + result.losses)
            logger.info(f"vs {opp_type:11s} W={result.wins} D={result.draws} "
                        f"L={result.losses}  ({result.wins / played:.0%} wins)  "
                        f"fitness={result.fitness:.3f}")

    else:
        logger.error("Specify --checkpoint or --tournament")
        return

    evaluator.shutdown()
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
