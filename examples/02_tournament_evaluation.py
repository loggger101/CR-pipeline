"""Example: Tournament Evaluation

Demonstrates evaluating a saved population in a tournament using the CURRENT
public API (FitnessEvaluator.run_tournament -- the same path ``crp tournament``
uses). Swiss is the default because it scales to full populations; round-robin
is O(N^2) and only sensible for small fields.

Requires a trained population checkpoint first:

    python examples/01_basic_training.py          # writes runs/example_basic/gen_XXXX/population.pt
    python examples/02_tournament_evaluation.py   # evaluates it
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.serialization import load_checkpoint
from src.train import FitnessEvaluator, TournamentFormat


def _latest_population_path(runs_dir: str = "runs/example_basic") -> Path | None:
    """Find the newest gen_XXXX/population.pt under a run directory."""
    base = Path(runs_dir)
    gens = sorted(base.glob("gen_*"), reverse=True) if base.exists() else []
    for g in gens:
        pop = g / "population.pt"
        if pop.exists():
            return pop
    # A bare runs/ dir may hold a top-level population.pt (older layouts).
    direct = Path(runs_dir) / "population.pt"
    return direct if direct.exists() else None


def _load_genomes(path: Path):
    """Extract the genome list from a population checkpoint."""
    payload = load_checkpoint(str(path))
    if isinstance(payload, dict):
        agents = payload.get("agents") or []
        weights_list = [np.asarray(a["weights"]) for a in agents]
        agent_ids = [a.get("agent_id", f"agent_{i}") for i, a in enumerate(agents)]
        return agent_ids, weights_list
    # Fallback: the checkpoint is a single genome array.
    return ["agent_0"], [np.asarray(payload).ravel()]


def run_tournament_evaluation():
    """Run a Swiss tournament over every agent in the newest population."""
    pop_path = _latest_population_path()
    if pop_path is None:
        raise SystemExit(
            "No population checkpoint found under runs/example_basic. "
            "Run examples/01_basic_training.py first."
        )

    agent_ids, weights_list = _load_genomes(pop_path)
    print(f"Loaded {len(agent_ids)} agents from {pop_path}")

    evaluator = FitnessEvaluator(num_workers=2, matches_per_agent=4)
    result = evaluator.run_tournament(
        agent_ids=agent_ids,
        weights_list=weights_list,
        format=TournamentFormat.SWISS,     # scales to full populations
        matches_per_pair=2,                # keep it fast for an example run
        seed=1234,
    )

    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS (Swiss)")
    print("=" * 60)
    for rank, (agent_id, score) in enumerate(result.rankings, 1):
        stats = result.agent_stats.get(agent_id)
        wld = ""
        if stats is not None:
            wld = f" ({stats.wins}W/{stats.draws}D/{stats.losses}L)"
        print(f"{rank:>2}. {agent_id:<16} score={score:.3f}{wld}")

    # ELO after the tournament (initial 1500 for everyone).
    if result.elo_ratings:
        top = max(result.elo_ratings, key=result.elo_ratings.get)
        print(f"\nHighest ELO: {top} at {result.elo_ratings[top]:.1f}")

    # Head-to-head matrix is available on the result for any pair.
    if len(agent_ids) >= 2 and (agent_ids[0], agent_ids[1]) in result.h2h_records:
        h2h = result.h2h_records[(agent_ids[0], agent_ids[1])]
        print(f"\nH2H {agent_ids[0]} vs {agent_ids[1]}: "
              f"{h2h.agent1_wins}-{h2h.agent2_wins} ({h2h.draws} draws)")

    return result


if __name__ == "__main__":
    run_tournament_evaluation()
