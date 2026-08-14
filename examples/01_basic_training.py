"""Example: Basic Evolutionary Training

Demonstrates:
- Configuring and running evolutionary training
- Monitoring training progress
- Saving checkpoints
- Evaluating results
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import EvolutionTrainer, TrainingConfig
from src.models import create_cnn_lstm_agent
from src.env.sim import ParallelRunner


def run_basic_training():
    """Run a basic evolutionary training run."""
    # Configure training
    config = TrainingConfig(
        population_size=100,
        elite_count=10,
        max_generations=50,
        crossover_rate=0.7,
        mutation_rate=0.05,
        mutation_std=0.1,
        num_workers=4,
        tournament_mode=True,
        tournament_format="round_robin",
        tournament_matches=4,
        seed=42,
        runs_dir="runs/example_basic",
    )

    # Initialize trainer
    trainer = EvolutionTrainer(config)

    # Run training
    print("Starting training...")
    trainer.train()

    # Print results
    print(f"Training complete!")
    print(f"Best fitness: {trainer.best_fitness:.4f}")
    print(f"Total generations: {trainer.generation}")
    print(f"Best agent: {trainer.best_agent}")

    return trainer


if __name__ == "__main__":
    trainer = run_basic_training()
