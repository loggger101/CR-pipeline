"""Example: Tournament Evaluation

Demonstrates:
- Running tournament-style evaluation
- Comparing multiple models
- ELO rating computation
- Head-to-head analysis
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import TournamentRunner, TournamentFormat
from src.viz.tournament_viz import compute_tournament_summary, render_bracket_ascii


def run_tournament_evaluation():
    """Run tournament evaluation on trained models."""
    # Configure tournament
    runner = TournamentRunner(
        checkpoint_dir="runs/example_basic",
        tournament_format=TournamentFormat.ROUND_ROBIN,
        matches_per_pair=10,
        num_workers=4,
    )

    # Run tournament
    print("Running tournament evaluation...")
    result = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print(result.summary())

    # Print bracket
    if result.bracket:
        print("\nBracket:")
        print(render_bracket_ascii(result.bracket))

    # Compute and display ELO
    elo_data = result.elo_history
    if elo_data:
        print("\nELO Progression:")
        for gen, ratings in elo_data[-5:]:
            print(f"  Gen {gen}: {ratings}")

    return result


if __name__ == "__main__":
    result = run_tournament_evaluation()
