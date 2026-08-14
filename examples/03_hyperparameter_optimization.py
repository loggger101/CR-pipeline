"""Example: Hyperparameter Optimization

Demonstrates:
- Bayesian optimization
- Grid search
- Random search
- Comparing optimization strategies
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import (
    BayesianOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    ParamSpace,
    ParamType,
    get_default_evolution_search_space,
)


def run_bayesian_optimization():
    """Run Bayesian hyperparameter optimization."""
    # Define search space
    space = get_default_evolution_search_space()

    # Initialize optimizer
    optimizer = BayesianOptimizer(
        param_space=space,
        n_trials=30,
        base_run_dir="runs/example_basic",
        output_dir="runs/hpo_bayesian",
    )

    # Run optimization
    print("Running Bayesian optimization...")
    result = optimizer.optimize()

    # Print results
    print("\nTop 5 configurations:")
    for i, trial in enumerate(result.top_trials(5)):
        print(f"  {i+1}. Fitness: {trial.best_fitness:.4f}")
        print(f"     Params: {trial.params}")

    return result


def run_grid_search():
    """Run grid search hyperparameter optimization."""
    space = {
        "crossover_rate": [0.5, 0.7, 0.9],
        "mutation_rate": [0.01, 0.05, 0.1],
        "mutation_std": [0.05, 0.1, 0.2],
    }

    optimizer = GridSearchOptimizer(
        param_space=space,
        base_run_dir="runs/example_basic",
        output_dir="runs/hpo_grid",
    )

    print("Running grid search...")
    result = optimizer.optimize()

    print(f"\nTotal configurations tested: {len(result.trials)}")
    print(f"Best fitness: {result.best_trial.best_fitness:.4f}")

    return result


if __name__ == "__main__":
    # Run Bayesian optimization (faster convergence)
    bayesian_result = run_bayesian_optimization()
    # Run grid search (exhaustive but slower)
    # grid_result = run_grid_search()
