#!/usr/bin/env python3
"""Launch tournament-based evolution training for CR-Pipeline.

Tournament-based evolution where agents battle each other head-to-head
to determine rankings, with tournament results driving the next generation.

Each generation:
1. Run tournament (round-robin, single elimination, etc.)
2. Use tournament rankings for selection pressure
3. Preserve top performers as elites
4. Evolve next generation using tournament-weighted crossover/mutation

Usage:
    python scripts/train_tournament.py [--config configs/evolution.yaml] [--population runs/best/checkpoint.pt]

Options:
    --config        Evolution config YAML file
    --population    Path to population checkpoint for seeding
    --tournament    Tournament format (round_robin, single_elim, double_elim, league)
    --matches       Matches per pair in tournament
    --workers       Number of parallel workers
    --max-gens      Maximum generations to train
    --seed          Random seed
    --runs-dir      Directory for training outputs
    --elite-count   Number of elite agents to preserve
    --verbose       Enable verbose logging
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import (
    FitnessEvaluator,
    TournamentResult,
    TournamentFormat,
    CheckpointManager,
    HyperparamsLoader,
)
from src.models import (
    Population,
    TournamentEvolutionStrategy,
)
from src.serialization import load_checkpoint


def load_population_weights(path: str) -> list:
    """Load weights from a checkpoint file."""
    checkpoint = load_checkpoint(path)
    if isinstance(checkpoint, dict) and "agents" in checkpoint:
        return [np.array(a["weights"]) for a in checkpoint["agents"]]
    elif isinstance(checkpoint, dict) and "weights" in checkpoint:
        return [np.array(checkpoint["weights"])]
    return [np.array(checkpoint)]


def save_checkpoint(
    population: Population,
    generation: int,
    best_fitness: float,
    best_agent: Optional[dict],
    tournament_history: List[dict],
    runs_dir: str,
) -> str:
    """Save a tournament checkpoint."""
    checkpoint_dir = Path(runs_dir) / "tournament_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "generation": generation,
        "best_fitness": best_fitness,
        "best_agent": best_agent,
        "tournament_history": tournament_history[-50:],  # Keep last 50
        "population_weights": [w.tolist() for w in population.get_population_weights()],
        "timestamp": time.time(),
    }

    checkpoint_path = checkpoint_dir / f"gen_{generation:04d}.pt"
    torch.save(checkpoint, str(checkpoint_path))
    logger.info(f"Saved tournament checkpoint to {checkpoint_path}")

    # Save latest
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, str(latest_path))

    return str(checkpoint_path)


def load_checkpoint(runs_dir: str) -> Optional[dict]:
    """Load the latest tournament checkpoint."""
    latest_path = Path(runs_dir) / "tournament_checkpoints" / "latest.pt"
    if latest_path.exists():
        return load_checkpoint(str(latest_path))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Train Clash Royale AI agents using tournament-based evolution."
    )
    parser.add_argument("--config", type=str, default="configs/evolution.yaml")
    parser.add_argument("--population", type=str, default=None)
    parser.add_argument(
        "--tournament",
        type=str,
        default="round_robin",
        choices=["round_robin", "single_elim", "double_elim", "league"],
        help="Tournament format",
    )
    parser.add_argument("--matches", type=int, default=4, help="Matches per pair")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--max-gens", type=int, default=100, help="Max generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--elite-count", type=int, default=10, help="Elite count")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("tournament_training.log")],
    )
    logger = logging.getLogger("train_tournament")

    logger.info("=" * 70)
    logger.info("CR-Pipeline: Tournament-Based Evolution Training")
    logger.info("=" * 70)

    # Load config
    loader = HyperparamsLoader()
    evolution_config = loader.load_evolution_config(args.config)

    # Determine tournament format
    format_map = {
        "round_robin": TournamentFormat.ROUND_ROBIN,
        "single_elim": TournamentFormat.SINGLE_ELIMINATION,
        "double_elim": TournamentFormat.DOUBLE_ELIMINATION,
        "league": TournamentFormat.LEAGUE,
    }
    tournament_format = format_map.get(args.tournament, TournamentFormat.ROUND_ROBIN)

    # Initialize population
    population_size = evolution_config.get("population", {}).get("size", 50)
    population = Population(population_size=population_size, elite_count=args.elite_count)

    # Initialize evaluator
    num_workers = args.workers or 4
    evaluator = FitnessEvaluator(num_workers=num_workers, matches_per_agent=args.matches)

    # Initialize tournament evolution strategy
    tournament_strategy = TournamentEvolutionStrategy(
        tournament_format=tournament_format,
        matches_per_pair=args.matches,
        elite_fraction=args.elite_count / population_size,
        crossover_rate=evolution_config.get("crossover", {}).get("rate", 0.7),
        mutation_rate=evolution_config.get("mutation", {}).get("rate", 0.05),
        mutation_std=evolution_config.get("mutation", {}).get("std", 0.1),
        seed=args.seed,
    )

    # Initialize checkpoint manager
    runs_path = Path(args.runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    checkpoint_mgr = CheckpointManager(str(runs_path))

    # Training state
    generation = 0
    best_fitness = -float("inf")
    best_agent: Optional[dict] = None
    tournament_history: List[dict] = []
    elo_history: Dict[str, List[float]] = {}  # agent_id -> list of ELO ratings

    # Resume from checkpoint if requested
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint:
            generation = checkpoint.get("generation", 0)
            best_fitness = checkpoint.get("best_fitness", -float("inf"))
            best_agent = checkpoint.get("best_agent")
            tournament_history = checkpoint.get("tournament_history", [])

            # Restore population
            pop_weights = checkpoint.get("population_weights", [])
            if pop_weights:
                population.set_population_weights([np.array(w) for w in pop_weights])
            logger.info(f"Resumed from checkpoint: gen={generation}, best_fitness={best_fitness}")
    elif args.population:
        elite_weights = load_population_weights(args.population)
        if elite_weights:
            # Use elite weights to seed population
            for i, w in enumerate(elite_weights[:population_size]):
                population.set_agent_weights(i, w)
            logger.info(f"Seeded population with {len(elite_weights)} elite weights")
    else:
        population.initialize(seed=args.seed)
        logger.info("Initialized new population")

    # Training loop
    start_time = time.time()
    max_gens = args.max_gens

    try:
        for gen in range(generation, max_gens):
            gen_start = time.time()
            logger.info(f"=== Generation {gen + 1} / {max_gens} ===")

            # Get current population weights
            weights_list = population.get_population_weights()
            n = len(weights_list)
            agent_ids = [f"agent_{i}" for i in range(n)]

            # Run tournament
            logger.info(f"Running {tournament_format.name} tournament...")
            tournament_result: TournamentResult = evaluator.run_tournament(
                agent_ids=agent_ids,
                weights_list=weights_list,
                format=tournament_format,
                matches_per_pair=args.matches,
                seed=args.seed + gen * 1000,
                generation=gen,
            )

            # Log tournament results
            logger.info(tournament_result.summary())

            # Track tournament info
            tournament_info = {
                "generation": gen,
                "rankings": [(aid, float(score)) for aid, score in tournament_result.rankings],
                "elo_ratings": {aid: float(elo) for aid, elo in tournament_result.elo_ratings.items()},
                "total_matches": tournament_result.total_matches,
                "agent_stats": {
                    aid: {
                        "wins": stats.wins,
                        "draws": stats.draws,
                        "losses": stats.losses,
                        "towers_destroyed": stats.towers_destroyed,
                        "total_duration": stats.total_duration,
                        "elo_rating": stats.elo_rating,
                        "win_rate": stats.win_rate,
                    }
                    for aid, stats in tournament_result.agent_stats.items()
                },
            }
            tournament_history.append(tournament_info)

            # Update ELO history
            for aid, elo in tournament_result.elo_ratings.items():
                if aid not in elo_history:
                    elo_history[aid] = []
                elo_history[aid].append(elo)

            # Update population fitness with tournament scores
            tournament_fitnesses = [score for _, score in tournament_result.rankings]
            population.evaluate(tournament_fitnesses)

            # Track best agent
            current_best = population.get_best()
            if current_best is not None and current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_agent = {
                    "weights": current_best.agent.get_weights().tolist(),
                    "fitness": current_best.fitness,
                    "generation": gen,
                }
                logger.info(f"New best fitness: {best_fitness:.3f}")

            # Evolve next generation using tournament strategy
            logger.info("Evolving next generation...")
            new_weights, evolution_info = tournament_strategy.evolve(
                population=weights_list,
                weights_list=weights_list,
                current_fitnesses=tournament_fitnesses,
                evaluator=evaluator,
                generation=gen,
            )
            population.set_population_weights(new_weights)

            # Save tournament summary
            gen_summary = tournament_strategy.get_tournament_summary()
            if gen_summary:
                logger.info(f"Tournament summary: {json.dumps(gen_summary, indent=2)}")

            # Checkpoint every 10 generations
            if (gen + 1) % 10 == 0:
                checkpoint_path = save_checkpoint(
                    population, gen + 1, best_fitness, best_agent,
                    tournament_history, args.runs_dir,
                )
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Save best agent
            if best_agent:
                checkpoint_mgr.save_best_agent(best_agent)

            elapsed = time.time() - gen_start
            logger.info(f"Generation {gen + 1} completed in {elapsed:.1f}s")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Final summary
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("Training complete!")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Final generation: {generation + max_gens}")
    logger.info(f"Best fitness: {best_fitness:.3f}")

    if best_agent:
        logger.info(f"Best agent saved to checkpoint")

    # Save final tournament history
    history_path = runs_path / "tournament_history.json"
    with open(history_path, "w") as f:
        json.dump(tournament_history, f, indent=2, default=str)
    logger.info(f"Tournament history saved to {history_path}")

    # Save ELO history for dashboard
    elo_path = runs_path / "elo_history.json"
    with open(elo_path, "w") as f:
        json.dump(elo_history, f, indent=2)
    logger.info(f"ELO history saved to {elo_path}")

    evaluator.shutdown()


if __name__ == "__main__":
    main()
