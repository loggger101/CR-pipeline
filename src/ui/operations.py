"""Pipeline operations the UI drives.

Kept free of Tk so each one can be exercised headlessly in tests: the widgets
only gather inputs, hand them to a function here, and render what comes back.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..env.sim.engine import SimulationEngine
from ..env.sim.parallel_runner import (
    OPPONENT_PROFILES, WorkerConfig, _OPPONENT_ACTIONS, _policy_action,
    _run_head_to_head, _run_matches,
)
from ..models.policy import DEFAULT_POLICY_SPEC
from ..serialization import load_agent_genome as _load_agent_genome
from ..train.trainer import EvolutionTrainer, TrainingConfig
from .arena_canvas import snapshot_from_engine
from .jobs import JobContext

TOURNAMENT_FORMATS = ["swiss", "round_robin", "single_elim", "double_elim", "league"]
SCRIPTED_OPPONENTS = sorted(_OPPONENT_ACTIONS)
MATCH_DURATIONS = ["short", "full", "overtime"]

# How a run's population starts.
START_FRESH = "Fresh population"
START_CONTINUE = "Continue a previous run"
START_SEED = "Start from chosen agents"
START_MODES = [START_FRESH, START_CONTINUE, START_SEED]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def resolve_runs_dir(project_root: str, runs_dir: Optional[str] = None) -> str:
    """Decide where runs live and make sure the folder exists.

    An explicit ``runs_dir`` is used verbatim; otherwise runs sit beside the
    project. Kept out of the window class so it can be tested without creating
    a Tk root (only one may exist per process).
    """
    target = Path(runs_dir) if runs_dir else Path(project_root) / "runs"
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def run_dir_for_resume(source: str) -> str:
    """The run directory a resume source belongs to.

    Continuing a run writes back into it, so its history, checkpoints and best
    agent stay in one place rather than being split across directories. The
    source may be the run folder, a ``gen_XXXX`` folder, or a population file.
    """
    path = Path(source)
    if path.is_file():
        path = path.parent
    if path.name.startswith("gen_"):
        path = path.parent
    return str(path)


def new_run_dir(runs_root: str, label: str = "run") -> str:
    """Allocate a fresh, timestamped directory for one training run.

    EvolutionTrainer writes ``gen_*/``, ``best/`` and ``fitness_history.json``
    straight into its ``runs_dir``, so pointing successive runs at the same
    folder makes them overwrite each other. One directory per run keeps runs
    listable and comparable.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = Path(runs_root)
    candidate = root / f"{label}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{label}_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def build_training_config(values: Dict[str, Any], runs_dir: str) -> TrainingConfig:
    """Turn UI field values into a validated :class:`TrainingConfig`.

    ``runs_dir`` is this run's own directory; see :func:`new_run_dir`.

    Raises:
        ValueError: with a message meant for display, if a field is unusable.
    """
    def as_int(key: str, minimum: int = 1) -> int:
        try:
            number = int(values[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key.replace('_', ' ')} must be a whole number")
        if number < minimum:
            raise ValueError(
                f"{key.replace('_', ' ')} must be at least {minimum}")
        return number

    population = as_int("population_size", 2)
    elite = as_int("elite_count", 1)
    if elite >= population:
        raise ValueError("elite count must be smaller than population size")

    fmt = str(values.get("tournament_format", "swiss")).lower()
    if fmt not in TOURNAMENT_FORMATS:
        raise ValueError(f"unknown tournament format {fmt!r}")

    opponent = str(values.get("opponent_type", "balanced"))
    if opponent not in SCRIPTED_OPPONENTS:
        raise ValueError(f"unknown opponent {opponent!r}")

    duration = str(values.get("match_duration", "short"))
    if duration not in MATCH_DURATIONS:
        raise ValueError(f"unknown match duration {duration!r}")

    # How the population starts: fresh, continuing a run, or seeded from
    # chosen agents.
    start_mode = str(values.get("start_mode", START_FRESH))
    if start_mode not in START_MODES:
        raise ValueError(f"unknown start mode {start_mode!r}")

    resume_from = None
    seed_agents = None
    additional = None
    generations = as_int("max_generations")

    if start_mode == START_CONTINUE:
        source = str(values.get("resume_from") or "").strip()
        if not source:
            raise ValueError("choose a run to continue")
        if EvolutionTrainer.find_population_checkpoint(source) is None:
            raise ValueError(
                f"no saved population found in {Path(source).name!r}; that run "
                f"has no checkpoints to continue from"
            )
        resume_from = source
        # Generations means "this many more" when continuing.
        additional = generations

    elif start_mode == START_SEED:
        agents = [p for p in (values.get("seed_agents") or []) if p]
        if not agents:
            raise ValueError("choose at least one agent to start from")
        for path in agents:
            load_agent_genome(path)   # fail now, with a clear message
        seed_agents = agents

    return TrainingConfig(
        population_size=population,
        elite_count=elite,
        max_generations=generations,
        additional_generations=additional,
        resume_from=resume_from,
        seed_agents=seed_agents,
        num_workers=as_int("num_workers"),
        seed=as_int("seed", 0),
        tournament_mode=bool(values.get("tournament_mode", True)),
        tournament_format=fmt,
        tournament_matches=as_int("tournament_matches"),
        hall_of_fame_size=as_int("hall_of_fame_size", 0),
        matches_per_agent=as_int("matches_per_agent"),
        opponent_type=opponent,
        match_duration=duration,
        runs_dir=runs_dir,
        curriculum_learning=False,
        diversity_preservation=False,
        # Checkpoint often enough that a stopped run is still worth
        # continuing. A quarter of the run meant a 200-generation job wrote
        # nothing until generation 50, so stopping at 11 left nothing to
        # resume from.
        checkpoint_interval=max(1, min(10, generations)),
    )


def run_training(ctx: JobContext, config: TrainingConfig) -> Dict[str, Any]:
    """Run a training job, streaming per-generation progress to ``ctx``.

    Stopping is cooperative: the trainer checks its ``running`` flag between
    generations, so a cancel takes effect once the current generation
    finishes rather than tearing down a tournament mid-flight.
    """
    trainer = EvolutionTrainer(config)
    history: List[dict] = []

    def on_generation(snapshot: dict) -> None:
        history.append(snapshot)
        ctx.progress(snapshot)
        if ctx.cancelled:
            trainer.stop()

    trainer.on_generation = on_generation
    started = time.time()

    try:
        ctx.log(f"Training started: {config.population_size} agents, "
                f"{config.max_generations} generations, "
                + ("tournament (" + config.tournament_format + ")"
                   if config.tournament_mode
                   else f"vs {config.opponent_type}"))
        trainer.train()
    finally:
        trainer.close()

    best_path = Path(config.runs_dir) / "best" / "best_agent.pt"
    return {
        "generations_completed": len(history),
        # In tournament mode the trainer ranks agents across generations by
        # ELO, so best_fitness carries an ELO rating rather than a fitness
        # score. The label travels with it so the UI cannot mislabel it.
        "best_score": trainer.best_fitness,
        "best_score_label": "ELO" if config.tournament_mode else "fitness",
        "best_generation": trainer.best_generation,
        "run_dir": config.runs_dir,
        "elapsed": time.time() - started,
        "best_agent_path": str(best_path) if best_path.exists() else None,
        "cancelled": ctx.cancelled,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Watching matches
# ---------------------------------------------------------------------------

@dataclass
class MatchRecording:
    """A finished match captured as replayable snapshots."""
    frames: List[dict]
    winner: str
    reason: str
    ticks: int

    def __len__(self) -> int:
        return len(self.frames)


def play_match(ctx: JobContext, genome: np.ndarray,
               opponent: str = "balanced",
               opponent_genome: Optional[np.ndarray] = None,
               seed: int = 7, capture_every: int = 4,
               match_duration_ticks: int = 1800) -> MatchRecording:
    """Run one match and capture it for playback.

    The match is simulated as fast as the CPU allows and recorded, rather than
    animated in real time; the UI then scrubs the recording at whatever speed
    the viewer wants. Capturing every tick would produce 1800 frames per match
    for no visible benefit, so frames are sampled.
    """
    engine = SimulationEngine(seed=seed, record_replay=False,
                              match_duration_ticks=match_duration_ticks)
    engine.reset()

    opponent_fn = _OPPONENT_ACTIONS.get(opponent)
    frames: List[dict] = [snapshot_from_engine(engine)]
    result = None
    max_ticks = match_duration_ticks + engine.overtime_ticks + 100

    while not engine.terminated and engine.tick <= max_ticks:
        if ctx.cancelled:
            break
        player_action = _policy_action(genome, engine, "player")
        if opponent_genome is not None:
            enemy_action = _policy_action(opponent_genome, engine, "opponent")
        elif opponent_fn is not None:
            enemy_action = opponent_fn(engine)
        else:
            enemy_action = None
        result = engine.step(player_action, enemy_action)
        if engine.tick % capture_every == 0:
            frames.append(snapshot_from_engine(engine))

    winner = (result.info.get("winner", "none") if result else "none")
    reason = (result.info.get("reason", "") if result else "")
    frames.append(snapshot_from_engine(engine, winner=winner, reason=reason))
    return MatchRecording(frames=frames, winner=winner, reason=reason,
                          ticks=engine.tick)


# ---------------------------------------------------------------------------
# Agent evaluation
# ---------------------------------------------------------------------------

# Re-exported so the UI keeps one import surface; the implementation is shared
# with the trainer, which seeds populations from the same files.
load_agent_genome = _load_agent_genome


def evaluate_against_scripted(ctx: JobContext, genome: np.ndarray,
                              opponents: List[str], matches: int = 6,
                              seed: int = 4242) -> List[Dict[str, Any]]:
    """Play ``genome`` against each named scripted opponent."""
    rows = []
    for opponent in opponents:
        if ctx.cancelled:
            break
        ctx.log(f"Playing {matches} matches vs {opponent}...")
        config = WorkerConfig(seed=seed, match_count=matches)
        result = _run_matches(0, config, genome, opponent, None)
        total = max(1, result.wins + result.draws + result.losses)
        rows.append({
            "opponent": opponent,
            "wins": result.wins,
            "draws": result.draws,
            "losses": result.losses,
            "win_rate": result.wins / total,
            "avg_towers": result.avg_towers_destroyed,
            "fitness": result.fitness,
        })
        ctx.progress(rows[-1])
    return rows


def evaluate_head_to_head(ctx: JobContext, genome_a: np.ndarray,
                          genome_b: np.ndarray, matches: int = 20,
                          seed: int = 777) -> Dict[str, Any]:
    """Play two saved agents against each other, sides swapped."""
    ctx.log(f"Playing {matches} matches, sides swapped...")
    result = _run_head_to_head(0, genome_a, genome_b,
                               matches_per_pair=matches, seed=seed)
    meta = result.metadata
    played = (meta["agent1_wins"] + meta["agent1_draws"] + meta["agent1_losses"])
    return {
        "a_wins": meta["agent1_wins"],
        "a_draws": meta["agent1_draws"],
        "a_losses": meta["agent1_losses"],
        "b_wins": meta["agent2_wins"],
        "matches": played,
        "a_win_rate": meta["agent1_wins"] / max(1, played),
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _looks_like_run(path: Path) -> bool:
    """Whether ``path`` is a training run directory.

    A run holds ``fitness_history.json`` and/or a ``best/`` folder. Generation
    checkpoints (``gen_0001``...) live *inside* a run and must not be listed as
    runs themselves.
    """
    if not path.is_dir() or path.name in ("best", "matches", "monitoring"):
        return False
    if path.name.startswith("gen_"):
        return False
    return ((path / "fitness_history.json").exists()
            or (path / "best").is_dir()
            or any(path.glob("gen_*")))


def list_runs(runs_dir: str) -> List[Dict[str, Any]]:
    """Summarise training runs found under ``runs_dir``.

    Reads the directory directly rather than going through RunManager so the
    UI still lists partial runs that lack a full metrics file.
    """
    root = Path(runs_dir)
    if not root.exists():
        return []

    runs = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not _looks_like_run(entry):
            continue
        summary: Dict[str, Any] = {
            "run_id": entry.name,
            "path": str(entry),
            "generations": 0,
            "best_fitness": None,
            "modified": entry.stat().st_mtime,
            "has_agent": (entry / "best" / "best_agent.pt").exists(),
        }
        history = _read_json(entry / "fitness_history.json")
        if isinstance(history, dict) and history.get("best"):
            summary["generations"] = len(history["best"])
            summary["best_fitness"] = max(history["best"])
            summary["history"] = history
        metadata = _read_json(entry / "best" / "metadata.json")
        if isinstance(metadata, dict):
            summary["best_generation"] = metadata.get("best_generation")
        runs.append(summary)
    return runs


def load_run_history(run_path: str) -> Dict[str, List[float]]:
    """Load a run's per-generation fitness series."""
    history = _read_json(Path(run_path) / "fitness_history.json")
    if isinstance(history, dict):
        return {k: v for k, v in history.items() if isinstance(v, list)}
    return {}


def _read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
