#!/usr/bin/env python3
"""Launch simulation training for CR-Pipeline.

Usage:
    python scripts/train_sim.py [--config configs/evolution.yaml] [--sim-config configs/sim_game.yaml] [--phase phase2_basic] [--resume runs/checkpoints/gen_0100/metadata.json]

Options:
    --config        Evolution config YAML file
    --sim-config    Simulation config YAML file
    --phase         Training phase preset (phase1_baseline through phase5_competitive)
    --resume        Path to resume from checkpoint
    --workers       Number of parallel workers
    --max-gens      Maximum generations to train
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import EvolutionTrainer, TrainingConfig, HyperparamsLoader
from src.env.sim import SimulationEngine


def main():
    parser = argparse.ArgumentParser(
        description="Train Clash Royale AI agents using evolutionary algorithms."
    )
    parser.add_argument("--config", type=str, default="configs/evolution.yaml",
                       help="Path to evolution config YAML")
    parser.add_argument("--sim-config", type=str, default="configs/sim_game.yaml",
                       help="Path to simulation config YAML")
    parser.add_argument("--phase", type=str, default=None,
                       choices=["phase1_baseline", "phase2_basic", "phase3_advanced",
                                "phase4_finetune", "phase5_competitive"],
                       help="Training phase preset")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers")
    parser.add_argument("--max-gens", type=int, default=None,
                       help="Maximum generations to train")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--runs-dir", type=str, default="runs",
                       help="Directory for training outputs")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )
    logger = logging.getLogger("train_sim")

    logger.info("=" * 60)
    logger.info("CR-Pipeline: Evolutionary Training")
    logger.info("=" * 60)

    # Load configs
    loader = HyperparamsLoader()

    # Load evolution config
    evolution_config = loader.load_evolution_config(args.config)
    logger.info(f"Loaded evolution config from {args.config}")

    # Load simulation config
    sim_config = loader.load_sim_config(args.sim_config)
    logger.info(f"Loaded simulation config from {args.sim_config}")

    # Apply phase preset if specified
    if args.phase:
        phase_config = loader.load_phase_config(args.phase)
        logger.info(f"Applied phase preset: {args.phase}")

        # Merge phase evolution overrides
        if "evolution" in phase_config:
            for key, value in phase_config["evolution"].items():
                evolution_config[key] = value

        # Merge phase sim overrides
        if "sim" in phase_config:
            for key, value in phase_config["sim"].items():
                sim_config[key] = value

    # Build training config
    training_config = TrainingConfig(
        population_size=evolution_config.get("population", {}).get("size", 200),
        elite_count=evolution_config.get("population", {}).get("elite_count", 10),
        elite_preservation=evolution_config.get("population", {}).get("elite_preservation", True),
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
        matches_per_agent=evolution_config.get("fitness", {}).get("matches_per_agent", 5),
        match_duration=evolution_config.get("fitness", {}).get("match_duration", "full"),
        scoring_weights=evolution_config.get("fitness", {}).get("scoring", {}),
        max_generations=evolution_config.get("training", {}).get("max_generations", 500),
        early_stopping_patience=evolution_config.get("training", {}).get(
            "early_stopping", {}).get("patience", 30),
        early_stopping_min_improvement=evolution_config.get("training", {}).get(
            "early_stopping", {}).get("min_improvement", 0.5),
        checkpoint_interval=evolution_config.get("checkpoint", {}).get("interval", 10),
        max_checkpoints=evolution_config.get("checkpoint", {}).get("max_checkpoints", 50),
        num_workers=args.workers or evolution_config.get("training", {}).get("parallel", {}).get("workers", 8),
        batch_size=evolution_config.get("training", {}).get("parallel", {}).get("batch_size", 50),
        timeout=evolution_config.get("training", {}).get("parallel", {}).get("timeout", 300),
        log_interval=evolution_config.get("training", {}).get("logging", {}).get("log_interval", 1),
        save_full_history=evolution_config.get("training", {}).get("logging", {}).get("save_full_history", True),
        runs_dir=args.runs_dir,
        resume_from=args.resume,
        seed=args.seed,
    )

    # Override with CLI args
    if args.max_gens:
        training_config.max_generations = args.max_gens

    # Print config summary
    logger.info("Training Configuration:")
    logger.info(f"  Population size: {training_config.population_size}")
    logger.info(f"  Elite count: {training_config.elite_count}")
    logger.info(f"  Selection: {training_config.selection_strategy}")
    logger.info(f"  Crossover: {training_config.crossover_strategy} (rate={training_config.crossover_rate})")
    logger.info(f"  Mutation: {training_config.mutation_strategy} (rate={training_config.mutation_rate}, std={training_config.mutation_std})")
    logger.info(f"  Matches per agent: {training_config.matches_per_agent}")
    logger.info(f"  Max generations: {training_config.max_generations}")
    logger.info(f"  Workers: {training_config.num_workers}")
    logger.info(f"  Seed: {training_config.seed}")

    # Create and run trainer
    trainer = EvolutionTrainer(training_config)

    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")
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
