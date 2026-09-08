"""Example: Hyperparameter Optimization

Demonstrates hyperparameter search over GA settings using the CURRENT public
API. Each trial runs a short, real training run and is scored by the best
fitness reached (ELO in tournament mode) -- exactly what ``crp hpo`` does, so a
ranking that holds under these reduced budgets tends to hold at scale.

Three optimizers share one interface: BayesianOptimizer (default), GridSearch-
Optimizer and RandomSearchOptimizer; all take an objective callable that maps a
parameter dict to a score.

    python examples/03_hyperparameter_optimization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

from src.train import (
    BayesianOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    EvolutionTrainer,
    TrainingConfig,
    get_default_evolution_search_space,
)


def make_objective(pop_cap: int = 16, gens: int = 2):
    """One short real training run per candidate configuration."""
    counter = {"n": 0}

    def objective(params: dict) -> float:
        counter["n"] += 1
        pop_size = max(4, min(int(params.get("population_size", pop_cap)), pop_cap))
        elite_count = max(1, min(int(params.get("elite_count", 4)), pop_size // 5 or 1))
        config = TrainingConfig(
            population_size=pop_size,
            elite_count=elite_count,
            crossover_rate=float(params.get("crossover_rate", 0.7)),
            mutation_rate=float(params.get("mutation_rate", 0.05)),
            mutation_std=float(params.get("mutation_std", 0.1)),
            max_generations=gens,
            # One subdirectory per trial so checkpoints never collide between
            # candidates (the same reason `crp hpo` uses a tempdir per trial).
            runs_dir=str(Path("runs") / "example_hpo" / f"trial_{counter['n']}"),
        )
        with EvolutionTrainer(config) as trainer:
            trainer.train()
        score = trainer.best_fitness
        return float(score if math.isfinite(score) else 0.0)

    return objective


def run_bayesian_optimization(n_trials: int = 6):
    """Bayesian search over the default evolution parameter space."""
    optimizer = BayesianOptimizer(
        param_spaces=get_default_evolution_search_space(),
        n_initial=3,
        n_iterations=n_trials,
        seed=42,
    )
    result = optimizer.optimize(make_objective(pop_cap=16, gens=2), verbose=True)

    print("\nTop configuration (Bayesian):")
    for key, value in sorted(result.best_params.items()):
        print(f"  {key}: {value}")
    print(f"Best score: {result.best_score:.3f} over {result.n_evaluations} evaluations "
          f"in {result.optimization_time:.0f}s")


def run_random_search(n_trials: int = 6):
    """Random search with the same budget -- a cheap baseline to beat."""
    optimizer = RandomSearchOptimizer(
        param_spaces=get_default_evolution_search_space(),
        n_trials=n_trials,
        seed=42,
    )
    result = optimizer.optimize(make_objective(pop_cap=16, gens=2), verbose=True)

    print("\nTop configuration (random search):")
    for key, value in sorted(result.best_params.items()):
        print(f"  {key}: {value}")
    print(f"Best score: {result.best_score:.3f} over {result.n_evaluations} evaluations")


def run_grid_search(grid_points: int = 9):
    """Grid search -- exhaustive but expensive; fine for small spaces."""
    optimizer = GridSearchOptimizer(
        param_spaces=get_default_evolution_search_space(),
        grid_points=grid_points,
    )
    result = optimizer.optimize(make_objective(pop_cap=16, gens=2), verbose=True)

    print("\nTop configuration (grid search):")
    for key, value in sorted(result.best_params.items()):
        print(f"  {key}: {value}")
    print(f"Best score: {result.best_score:.3f} over {result.n_evaluations} evaluations")


if __name__ == "__main__":
    # Bayesian first (best samples per evaluation); the others are baselines.
    run_bayesian_optimization()
