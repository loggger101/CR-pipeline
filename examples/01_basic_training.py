"""Example: Basic Evolutionary Training

Demonstrates configuring a small training run with the CURRENT public API and
reading its results through the trainer's attributes. Mirrors what ``crp train``
does internally, so anything here that works will also work in your own scripts.

Keep the budget tiny when experimenting -- each generation plays population x
matches per agent, which is where all the time goes:

    python examples/01_basic_training.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import EvolutionTrainer, TrainingConfig


def run_basic_training():
    """Run a small tournament self-play training loop."""
    config = TrainingConfig(
        population_size=16,          # agents per generation
        elite_count=4,               # top agents copied forward unchanged
        max_generations=3,           # keep it short for an example run
        crossover_rate=0.7,
        mutation_rate=0.05,
        mutation_std=0.1,
        num_workers=2,
        tournament_mode=True,        # default; --no-tournament equivalent is False
        seed=42,
        runs_dir="runs/example_basic",
    )

    with EvolutionTrainer(config) as trainer:   # context manager detaches the run log handler
        trainer.train()

    print("Training complete!")
    print(f"Generations completed : {trainer.generation}")
    print(f"Best fitness (ELO)    : {trainer.best_fitness:.1f}")
    if trainer.best_agent is not None:
        print(f"Best agent            : {trainer.best_agent.agent_id} "
              f"(born gen {trainer.best_agent.generation_born})")

    return trainer


if __name__ == "__main__":
    run_basic_training()
