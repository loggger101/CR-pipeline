#!/usr/bin/env python3
"""Launch self-play training for CR-Pipeline.

Self-play training where agents evolve by playing against each other.
This is the most effective way to push agents to competitive levels.

Usage:
    python scripts/train_self_play.py [--config configs/evolution.yaml] [--population runs/best/checkpoint.pt]

Options:
    --config        Evolution config YAML file
    --population    Path to population checkpoint for seeding
    --elite-count   Number of elite agents to use as opponents
    --workers       Number of parallel workers
    --max-gens      Maximum generations to train
    --seed          Random seed
    --runs-dir      Directory for training outputs
    --verbose       Enable verbose logging
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import EvolutionTrainer, TrainingConfig, HyperparamsLoader, PHASE_CONFIGS
from src.env.sim import SimulationEngine
import numpy as np
import torch


def load_population_weights(path: str) -> list:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "agents" in checkpoint:
        return [np.array(a["weights"]) for a in checkpoint["agents"]]
    elif isinstance(checkpoint, dict) and "weights" in checkpoint:
        return [np.array(checkpoint["weights"])]
    return [np.array(checkpoint)]


def main():
    parser = argparse.ArgumentParser(
        description="Train Clash Royale AI agents using self-play evolution."
    )
    parser.add_argument("--config", type=str, default="configs/evolution.yaml")
    parser.add_argument("--population", type=str, default=None)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-gens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=str, default="runs")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--phase", type=str, default="phase5_competitive",
                       choices=["phase1_baseline", "phase2_basic", "phase3_advanced",
                                "phase4_finetune", "phase5_competitive"])
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                       format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                       handlers=[logging.StreamHandler(), logging.FileHandler("training.log")])
    logger = logging.getLogger("train_self_play")

    logger.info("=" * 60)
    logger.info("CR-Pipeline: Self-Play Training")
    logger.info("=" * 60)

    loader = HyperparamsLoader()
    evolution_config = loader.load_evolution_config(args.config)
    sim_config = loader.load_sim_config("configs/sim_game.yaml")
    phase_config = loader.load_phase_config(args.phase)

    if "evolution" in phase_config:
        for key, value in phase_config["evolution"].items():
            evolution_config[key] = value

    elite_weights = []
    if args.population:
        elite_weights = load_population_weights(args.population)

    training_config = TrainingConfig(
        population_size=evolution_config.get("population", {}).get("size", 200),
        elite_count=args.elite_count, elite_preservation=True,
        selection_strategy=evolution_config.get("selection", {}).get("strategy", "tournament"),
        tournament_size=evolution_config.get("selection", {}).get("tournament_size", 5),
        rank_weight=evolution_config.get("selection", {}).get("rank_weight", 1.5),
        crossover_strategy=evolution_config.get("crossover", {}).get("strategy", "blend"),
        crossover_rate=evolution_config.get("crossover", {}).get("rate", 0.7),
        blend_alpha=evolution_config.get("crossover", {}).get("blend_alpha", 0.5),
        mutation_strategy=evolution_config.get("mutation", {}).get("strategy", "gaussian"),
        mutation_rate=evolution_config.get("mutation", {}).get("rate", 0.05),
        mutation_std=evolution_config.get("mutation", {}).get("std", 0.1),
        min_mutation_std=evolution_config.get("mutation", {}).get("min_std", 0.01),
        max_mutation_std=evolution_config.get("mutation", {}).get("max_std", 0.5),
        adaptive_mutation=evolution_config.get("mutation", {}).get("adaptive", False),
        matches_per_agent=evolution_config.get("fitness", {}).get("matches_per_agent", 10),
        match_duration=evolution_config.get("fitness", {}).get("match_duration", "full"),
        scoring_weights=evolution_config.get("fitness", {}).get("scoring", {}),
        max_generations=evolution_config.get("training", {}).get("max_generations", 500),
        early_stopping_patience=evolution_config.get("training", {}).get("early_stopping", {}).get("patience", 30),
        early_stopping_min_improvement=evolution_config.get("training", {}).get("early_stopping", {}).get("min_improvement", 0.5),
        checkpoint_interval=evolution_config.get("checkpoint", {}).get("interval", 10),
        max_checkpoints=evolution_config.get("checkpoint", {}).get("max_checkpoints", 50),
        num_workers=args.workers or evolution_config.get("training", {}).get("parallel", {}).get("workers", 8),
        batch_size=evolution_config.get("training", {}).get("parallel", {}).get("batch_size", 50),
        timeout=evolution_config.get("training", {}).get("parallel", {}).get("timeout", 300),
        log_interval=evolution_config.get("training", {}).get("logging", {}).get("log_interval", 1),
        save_full_history=evolution_config.get("training", {}).get("logging", {}).get("save_full_history", True),
        runs_dir=args.runs_dir, seed=args.seed,
        diversity_preservation=True, diversity_threshold=0.5,
        curriculum_learning=True, use_training_decks=True)

    if args.max_gens:
        training_config.max_generations = args.max_gens

    logger.info("Self-Play Training Configuration:")
    logger.info(f"  Phase: {args.phase}")
    logger.info(f"  Population size: {training_config.population_size}")
    logger.info(f"  Elite count: {training_config.elite_count}")
    logger.info(f"  Selection: {training_config.selection_strategy}")
    logger.info(f"  Crossover: {training_config.crossover_strategy} (rate={training_config.crossover_rate})")
    logger.info(f"  Mutation: {training_config.mutation_strategy} (rate={training_config.mutation_rate}, std={training_config.mutation_std})")
    logger.info(f"  Matches per agent: {training_config.matches_per_agent}")
    logger.info(f"  Max generations: {training_config.max_generations}")
    logger.info(f"  Workers: {training_config.num_workers}")
    logger.info(f"  Elite opponents: {len(elite_weights)} agents")
    logger.info(f"  Seed: {training_config.seed}")

    trainer = EvolutionTrainer(training_config)
    try:
        logger.info("Starting self-play training...")
        logger.info("Agents will evolve by playing against each other.")
        trainer.train()
        logger.info("Self-play training complete!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        trainer.stop()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    finally:
        trainer.runner.shutdown()


if __name__ == "__main__":
    main()
