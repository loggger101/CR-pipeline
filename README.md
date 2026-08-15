# CR-Pipeline: Evolutionary Neural Network for Clash Royale

> An end-to-end pipeline for training Clash Royale AI agents using genetic/evolutionary algorithms — from live-game interaction to parallel simulation training, self-play evolution, tournament evaluation, and live visualization.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CR-Pipeline Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  Config      │───▶│  Pipeline    │───▶│  Training    │                 │
│  │  System      │    │  Orchestr.   │    │  Loop        │                 │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐                 │
│  │  Data        │───▶│  Sim Engine  │───▶│  Evolution   │                 │
│  │  Augment.    │    │  (Parallel)  │    │  Strategy    │                 │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐                 │
│  │  CLI         │───▶│  Viz/Dash.   │◀───│  Experiment  │                 │
│  │  (crp)       │    │  (Streamlit) │    │  Tracking    │                 │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐                 │
│  │  Deploy      │◀───│  Model       │◀───│  Arch Search │                 │
│  │  / Export    │    │  Ensemble    │    │  & Registry  │                 │
│  └──────────────┘    └──────────────┘    └──────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Goal

Train a population of neural networks that can play Clash Royale competitively using evolutionary strategies. The pipeline supports:

- **Live-game interaction** — a model plays the real game by interpreting the screen and issuing commands.
- **Low-fidelity simulation** — a simplified game engine allows thousands of parallel training instances.
- **Self-play evolution** — agents evolve by playing against each other in tournament-style competitions.
- **Tournament evaluation** — round-robin, single/double elimination, and league formats with ELO tracking.
- **Neural Architecture Search** — automatically discover optimal network topologies.
- **Model Ensembling** — combine top models for improved performance.
- **Hyperparameter Optimization** — Bayesian, grid, random, and population-based search.
- **Live visualization** — real-time Streamlit dashboard with comprehensive analytics.
- **Curriculum learning** — automatic phase transitions based on convergence detection.
- **Data augmentation** — deck composition, opponent strategy, and game condition variation.
- **Model export** — ONNX, TorchScript, NumPy, JSON, and pickle formats.
- **Experiment tracking** — MLflow-like tracking with run comparison and report generation.

---

## 🏗️ Project Structure

```
CR-Pipeline/
├── src/
│   ├── env/                      # Game environments
│   │   ├── live/                 # Live-game interaction
│   │   │   ├── screen_capture.py     # Screen capture & preprocessing
│   │   │   ├── game_state.py         # State extraction from frames
│   │   │   └── action_mapper.py      # Output → in-game actions
│   │   └── sim/                  # Low-fidelity simulation engine
│   │       ├── engine.py             # Core game loop
│   │       ├── entities.py           # 30+ cards, towers, units
│   │       ├── actions.py            # Discrete action space
│   │       ├── state.py              # Game state representation
│   │       ├── rendering.py          # Arena visualization
│   │       └── parallel_runner.py    # Multi-process parallelism
│   ├── models/                   # Neural networks & evolution
│   │   ├── policy.py               # ★ The evolved policy: features + NumPy MLP
│   │   ├── architecture.py         # CNN+LSTM, CNN+MLP, CNN+ResNet, CNN+Transformer
│   │   ├── agent.py                # Agent wrapper (Torch net built lazily)
│   │   ├── population.py           # Population management & fitness tracking
│   │   ├── evolution.py            # GA operators (selection, crossover, mutation)
│   │   ├── architecture_search.py  # Neural architecture search
│   │   └── ensemble.py             # Model ensembling methods
│   ├── train/                    # Training loop & management
│   │   ├── trainer.py              # Main evolution loop orchestrator
│   │   ├── evaluator.py            # Fitness evaluation & tournament runner
│   │   ├── checkpoint.py           # Save/load training state
│   │   ├── hyperparams.py          # Config management with phase presets
│   │   ├── hpo.py                  # Hyperparameter optimization
│   │   ├── experiment_tracking.py  # MLflow-like experiment tracking
│   │   └── pipeline.py             # Pipeline orchestration with DAG
│   ├── viz/                      # Visualization & reporting
│   │   ├── dashboard.py            # Advanced Streamlit dashboard (6 tabs)
│   │   ├── metrics.py              # Advanced metrics (growth, acceleration)
│   │   ├── runs_manager.py         # Run discovery, comparison, stats
│   │   ├── tournament_viz.py       # ELO, brackets, H2H charts
│   │   ├── reports.py              # HTML, Markdown, JSON report generation
│   │   ├── replay.py               # Replay viewer
│   │   ├── live_game_view.py       # Live gameplay overlay
│   │   └── rendering.py            # Simulation arena renderer
│   ├── ui/                       # Desktop application (Tkinter)
│   │   ├── app.py                  # Window and the four tabs
│   │   ├── operations.py           # Pipeline actions the UI drives
│   │   ├── jobs.py                 # Background job runner + event queue
│   │   ├── arena_canvas.py         # Arena drawn on a Tk canvas
│   │   └── chart.py                # Embedded matplotlib chart
│   ├── deploy/                   # Model export & deployment
│   │   └── export.py              # ONNX, TorchScript, NumPy, JSON export
│   ├── config/                   # Configuration system
│   │   ├── validation.py          # Schema-based validation & templating
│   │   └── __init__.py            # Package exports
│   └── data/                     # Data augmentation
│       ├── augmentation.py        # Deck, strategy, condition augmentation
│       └── __init__.py            # Package exports
├── configs/                      # YAML configuration files
│   ├── evolution.yaml            # Evolution hyperparameters
│   ├── sim_game.yaml             # Simulation engine config
│   └── live_game.yaml            # Live-game interaction config
├── packaging/                    # Standalone executable build
│   ├── crp_gui.spec              # PyInstaller spec
│   └── build_exe.py              # Build script
├── scripts/                      # Entry points
│   ├── crp_gui.py                # Desktop app launcher
│   ├── crp.py                    # Unified CLI (11 commands)
│   ├── train_sim.py              # Simulation training
│   ├── train_self_play.py        # Self-play evolution
│   ├── train_tournament.py       # Tournament evaluation
│   ├── play_live.py              # Live-game agent
│   ├── launch_dashboard.py       # Streamlit dashboard
│   └── evaluate.py               # Agent evaluation
├── tests/                        # Test suite (279 tests)
│   ├── test_sim_engine.py        # Engine tests
│   ├── test_sim_regressions.py   # ★ Guards previously-silent engine defects
│   ├── test_training_signal.py   # ★ Guards that training actually learns
│   ├── test_policy.py            # Evolved policy: packing, forward, features
│   ├── test_evolution.py         # Evolution strategy tests
│   ├── test_agent.py             # Agent tests
│   ├── test_tournament.py        # Tournament tests
│   ├── test_tournament_viz.py    # Visualization tests
│   ├── test_advanced_metrics.py  # Metrics tests
│   └── test_integration.py       # End-to-end integration tests
├── assets/                       # Static assets
│   ├── card_data.json            # 97+ card definitions (Level 11 stats)
│   └── maps/                     # Arena layouts
├── runs/                         # Training outputs (generated)
│   ├── <run_id>/                 # Per-run directory
│   │   ├── metrics.json          # Training metrics
│   │   ├── fitness_history.json  # Per-generation fitness
│   │   ├── config.yaml           # Run configuration
│   │   └── checkpoints/          # Checkpoint snapshots
│   └── best/                     # Best agent snapshot
├── reports/                      # Generated reports
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── Dockerfile                    # Container definition
├── pyproject.toml                # Project metadata
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 📦 Installation

### Option 1: pip (Recommended)

```bash
git clone https://github.com/loggger101/CR-pipeline.git
cd CR-pipeline
pip install -r requirements.txt
```

### Option 2: Conda

```bash
git clone https://github.com/loggger101/CR-pipeline.git
cd CR-pipeline
conda env create -f environment.yml
conda activate cr-pipeline
```

### Option 3: Docker

```bash
git clone https://github.com/loggger101/CR-pipeline.git
cd CR-pipeline
docker build -t cr-pipeline .
docker run -it --gpus all cr-pipeline
```

### GPU Support

For GPU acceleration, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 🖥️ Desktop App

A native window for driving the pipeline — no terminal, no browser.

```bash
python scripts/crp_gui.py
```

| Tab | What it does |
|---|---|
| **Train** | Set population, generations, tournament format, seed and workers. Start/stop a run and watch generation, best/mean fitness, champion ELO and record update live on a chart. |
| **Watch** | Load an agent and watch it play a match on the arena — towers, troops, elixir and crowns — with play/pause, scrubbing and a speed control. |
| **Runs** | Every past run with its generation count and best score. Select two or more to compare their fitness curves. |
| **Agents** | Load a saved agent and play it against all five scripted baselines, or head-to-head against a second agent. |

Each run gets its own timestamped folder under `runs/`, and a finished run
hands its best agent straight to the Watch and Agents tabs.

### Continuing from earlier work

The **Start from** control at the top of the Train tab decides where a run's
population comes from:

| Start from | What happens |
|---|---|
| **Fresh population** | Random genomes, as usual. |
| **Continue a previous run** | Picks up exactly where that run stopped — same population, generation counter, hall of fame and ELO ratings. "Generations" becomes *extra* generations, and the run writes back into its own folder so its history stays in one piece. |
| **Start from chosen agents** | Pick one or more `.pt` agents — from an earlier run or copied in from anywhere. Those genomes go into the population unchanged and the remaining slots are filled with mutated copies, so nothing already learned is thrown away but there is still variation to select on. |

The same thing from Python:

```python
# Continue a run for 50 more generations
config = TrainingConfig(runs_dir="runs/run_20260814_120000",
                        resume_from="runs/run_20260814_120000",
                        additional_generations=50)

# Or start a new run from agents you liked
config = TrainingConfig(runs_dir="runs/my_new_run",
                        seed_agents=["runs/run_a/best/best_agent.pt",
                                     "downloaded_champion.pt"],
                        seed_mutation_std=0.08,
                        max_generations=100)
```

`resume_from` accepts a run directory, a `gen_XXXX` folder, or a
`population.pt` file — the latest checkpoint is used when given a directory.
Resuming needs a checkpoint, so keep `checkpoint_interval` at a value that
actually fires during your run.

Training runs on a worker thread, so the window stays responsive; **Stop** ends
the run cleanly after the current generation rather than killing it mid-tournament.

### Building a standalone executable

```bash
pip install pyinstaller
python packaging/build_exe.py --clean
```

Produces `CR-Pipeline/CR-Pipeline.exe`. Double-click it — runs are saved to a
`runs` folder beside the executable. The build takes several minutes.

**Where it builds.** Normally into `dist/` in the project. If the project sits
in a cloud-synced folder (OneDrive, Dropbox), the build moves to
`%LOCALAPPDATA%\CR-Pipeline-build` instead — a multi-gigabyte bundle written
into a synced folder gets uploaded, and the sync client locks files mid-build,
which surfaces as a baffling `PermissionError` partway through. Override with
`--output PATH`.

**On bundle size.** With a CUDA build of PyTorch the bundle is around 4 GB,
almost all of it GPU libraries. The app only uses torch to read and write
checkpoints — the evolved policy is pure NumPy — so building inside a venv with
the CPU-only wheel cuts it dramatically:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`build_exe.py` detects a CUDA install and says so before building.

**Debugging a frozen build.** A windowed build reports startup failures in a
message box you cannot capture. Set `CRP_CONSOLE=1` before building to get a
console variant that prints the traceback instead.

The frozen entry point calls `multiprocessing.freeze_support()` before anything
else. Without that, every worker the training pool spawns re-executes the
`.exe` from the top and opens another window, without end.

If you would rather not build anything, `CR-Pipeline.bat` opens the same app
using the Python already installed on the machine.

---

## 🚀 Quick Start

### Using the CLI (Recommended)

```bash
# Train agents in simulation
crp train --max-gens 100 --population-size 200 --workers 8

# Run tournament evaluation on top agents
crp tournament --format round_robin --matches 4 --run-dir runs

# Run hyperparameter optimization
crp hpo --optimizer bayesian --trials 30 --base-run run_123

# Launch the advanced visualization dashboard
crp dashboard --runs-dir runs

# Generate a training report
crp report --run-dir runs/run_123 --type training

# Compare two training runs
crp compare --runs run_123 run_456

# Export best model
crp export --run-dir runs/run_123 --formats torch,onnx,numpy

# List all experiments
crp experiments --list

# Run architecture search
crp search --generations 20 --population 20

# Benchmark a model
crp benchmark --model runs/run_123/best_agent.pt --input-shape 1,8,6,16
```

### Using Python Scripts Directly

```python
# 1. Initialize training
from src.train import EvolutionTrainer, TrainingConfig

config = TrainingConfig(
    population_size=200,
    elite_count=10,
    max_generations=500,
    crossover_rate=0.7,
    mutation_rate=0.05,
    mutation_std=0.1,
    num_workers=8,
    tournament_mode=True,
    tournament_format="round_robin",
    tournament_matches=4,
    seed=42,
)

trainer = EvolutionTrainer(config)
trainer.train()

# 2. Run tournament evaluation
from src.train import TournamentRunner, TournamentFormat

runner = TournamentRunner(
    checkpoint_dir="runs/best",
    tournament_format=TournamentFormat.ROUND_ROBIN,
    matches_per_pair=4,
    num_workers=4,
)
result = runner.run()
print(result.summary())

# 3. Launch dashboard
from src.viz.dashboard import run_advanced_dashboard
run_advanced_dashboard(runs_dir="runs")
```

---

## 🏆 Tournament Training (the main loop)

**Agents are trained by playing each other.** Fitness is a competitor's
standing in a tournament against the rest of the population plus the hall of
fame — not a score against scripted bots. This is the default
(`tournament_mode=True`).

### Each generation

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. Entrants = whole population + hall of fame (past champions)   │
│ 2. Swiss pairing: every entrant plays one matchup per round,     │
│    against an opponent on a similar score                        │
│ 3. Each matchup is side-swapped and run across the worker pool   │
│ 4. Standings -> fitness;  ELO carries forward between generations│
│ 5. Selection, crossover and mutation act on those standings      │
│ 6. This generation's champion joins the hall of fame             │
└──────────────────────────────────────────────────────────────────┘
```

### Why Swiss

Round-robin is O(N²), which does not survive contact with a real population
size. Swiss ranks the same field in `N·R/2` matchups with `R = ⌈log₂ N⌉`:

| Population | Swiss rounds | Swiss matchups | Round-robin matchups |
|---|---|---|---|
| 8 | 3 | 12 | 28 |
| 24 | 5 | 60 | 276 |
| 64 | 6 | 192 | 2,016 |
| 200 | 8 | 800 | 19,900 |

At 16 agents, Swiss reproduces the full round-robin ranking with a rank
correlation of **+0.83** at ~5× less compute — and every entrant plays the same
number of games, so standings stay comparable.

### Reading progress: watch ratings, not fitness

Tournament fitness is points per match inside a closed field, so **its mean sits
near 0.5 no matter how strong the population gets**. A flat fitness curve in
tournament mode is arithmetic, not a stalled run.

What moves is rating against the hall of fame. As the population outgrows its
former champions, those champions' ELO falls relative to the field — in one
real 61-generation run, `hof_gen0` drifted from 1483 down to 1441 while mean
fitness barely changed. The Train tab charts champion ELO and past-champion
ELO for exactly this reason, and progress snapshots carry `population_elo` and
`hall_of_fame_elo`.

### The hall of fame anchors progress

Tournament fitness is *relative to the current field*. On its own it can rise
while nothing actually improves — the whole population can drift, or cycle
against its own meta, and every agent still looks better than its peers.

Two things prevent that:

- Past champions enter every tournament, so each generation is measured against
  fixed reference opponents rather than only against itself.
- **The best agent across a run is tracked by ELO, not by fitness.** Fitness
  from generation 2 and generation 9 are standings in different fields and are
  not comparable; ELO is, because the hall of fame is common to both.

`tests/test_tournament_training.py` closes the loop with an absolute check: the
final champion must beat the generation-0 champion head to head. That cannot be
satisfied by relative drift.

### Other formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **Swiss** *(default)* | Score-based pairing, `⌈log₂ N⌉` rounds | Training loop; scales to large populations |
| **Round Robin** | All agents play each other | Exhaustive ranking of a small field |
| **Single Elimination** | Loser exits, winner advances | Fast determination of a single winner |
| **Double Elimination** | Two losses to eliminate | More robust than single elimination |
| **League** | Promotion/relegation brackets | Long-term competitive evolution |

```
ELO update (per matchup, K = 32):
  expected = 1 / (1 + 10^((ELO_opponent - ELO_agent) / 400))
  ELO_new  = ELO_old + K × (actual - expected)
  where actual is the agent's share of the points actually won.
```

### Configuring it

```python
from src.train import EvolutionTrainer, TrainingConfig

config = TrainingConfig(
    population_size=64,
    tournament_mode=True,        # default
    tournament_format="swiss",   # default
    tournament_matches=2,        # matches per pairing, side-swapped
    tournament_rounds=None,      # None -> ceil(log2(entrants))
    hall_of_fame_size=4,         # past champions kept as benchmarks
    seed=42,
)
with EvolutionTrainer(config) as trainer:
    trainer.train()
```

Set `tournament_mode=False` to fall back to scoring against scripted opponents
(`opponent_type`), which is useful for calibration since those opponents never
change.

### Dashboard Tournament Tab

The Streamlit dashboard includes a dedicated tournament tab showing:
- **ELO progression** over generations (line chart)
- **Win rate** trends per agent (bar chart)
- **Head-to-head matrix** with win rates (heatmap)
- **Bracket visualization** for elimination formats
- **Detailed agent statistics** table

---

## 🧬 Evolution Strategy Details

### What Actually Evolves

The genetic algorithm optimises a **compact policy** (`src/models/policy.py`),
not the deep Torch networks. This is deliberate:

| | Evolved policy | Torch architectures |
|---|---|---|
| **Parameters** | 2,311 | ~9.3M (CNN+LSTM) |
| **Used for** | Every match in training | Architecture search, export, ensembling |
| **Inference** | Pure NumPy, no allocation per tick | Torch forward pass |

Genomes are plain float vectors, so they ship to worker processes cheaply and
evaluate thousands of times per match without rebuilding a model.

```
observation (64 features)  ->  tanh(32 hidden)  ->  5 card logits + 2 placement
```

The 64 features cover elixir and crown state, per-slot hand affordability and
card kind, all six tower healths, a pooled 3x4 troop-density grid per side, and
per-lane troop/HP summaries. Features are encoded from the acting player's
point of view — the arena is mirrored for the opponent — so a single genome can
play either side, which self-play and tournaments rely on.

`PolicySpec.num_params` is the contract between `Population` (which creates and
mutates genomes) and the parallel runner (which executes them). Changing the
spec in one place without the other raises a `ValueError` rather than silently
misreading the vector.

### Evaluation: common random numbers

Every agent in a generation is scored on the **same match seeds and the same
sequence of opponent decks**. Fitness comparison is therefore paired: the
difference between two agents reflects how they played, not which one drew the
kinder matchups. The shared seed advances per generation, so the population is
never graded repeatedly on one fixed set of games.

This matters more than it sounds. Against a competent opponent a single match
is close to a coin flip, so scoring each agent on its own seeds leaves
selection sorting mostly noise.

Evaluation is deterministic given `(genome, seed)`: exploration belongs to the
evolutionary operators, not the evaluator.

### Reproducibility

`EvolutionConfig.seed` drives selection, crossover, and mutation from one
stream, and `TrainingConfig.seed` is plumbed through to it. A seeded run
reproduces exactly; leave the seed as `None` for non-deterministic behaviour.

### Training baselines

The scripted opponents in `parallel_runner.py` share one heuristic core
(`_heuristic_opponent_action`) with per-personality knobs (`OpponentProfile`).
Each one answers pushes that cross the river by placing a counter — chosen for
staying power and damage per elixir, and required to be able to hit air or
ground as the threat demands — between the threat and its towers. When
unthreatened they bank elixir and commit at the bridge in the lane whose
defending tower is weakest, and they will answer a clump of troops with a
damage spell.

| Opponent | Role | Untrained genome win rate |
|---|---|---|
| `random` | Weak control | ~43% |
| `greedy` | Elixir dumper | ~43% |
| `aggressive` | Early commitment, late defence | ~30% |
| `balanced` | Defends, then counter-attacks | ~25% |
| `defensive` | Cheap answers, attacks on a full bank | ~25% |

Baseline strength sets the ceiling on what training can learn, so
`tests/test_opponents.py` asserts these stay in a band where there is headroom
in both directions.

### Network Architecture Options

| Architecture | Input | Hidden | Output | Best For |
|-------------|-------|--------|--------|----------|
| **CNN+LSTM** | Frames [B,F,C,H,W] | LSTM(256, 2L) → Dense(128) | Action logits | Temporal dynamics |
| **CNN+MLP** | Frames + state features | Two-stream → Concat(256) | Action logits | Fast training |
| **CNN+ResNet** | Single frame [B,C,H,W] | ResBlocks → LSTM | Action heads | Deep feature extraction |
| **CNN+Transformer** | Frame patches | CNN → Transformer → LSTM | Action logits | Global attention |

### Genetic Algorithm Operators

**Selection:** Tournament, Rank-based, Roulette wheel, Tournament-elite

**Crossover:** Blend (BLX-α), Single-point, Uniform, Arithmetic

**Mutation:** Gaussian, Uniform, Adaptive (increases when fitness plateaus)

**Elitism:** Top N agents preserved unchanged each generation

### Training Phases (Curriculum Learning)

| Phase | Description | Generations | Goal |
|-------|-------------|-------------|------|
| **Phase 1** | Random policy baseline | 10 | Verify pipeline works |
| **Phase 2** | Evolution in simplified arena | 50 | Learn basic card placement |
| **Phase 3** | Evolution in full simulation | 100 | Learn deck synergy & timing |
| **Phase 4** | Fine-tune on live game | 100 | Adapt to real-game noise |
| **Phase 5** | Self-play competitive evolution | 200 | Push to competitive level |

Automatic phase transitions occur when fitness variance drops below threshold.

---

## 📈 Visualization & Analytics

### Streamlit Dashboard (8 Tabs)

| Tab | Content |
|-----|---------|
| **Fitness** | Fitness curves, population stats, progress bars |
| **Statistics** | Growth rate, acceleration, convergence detection, Pareto front |
| **Tournament** | ELO progression, win rates, H2H matrix, brackets |
| **Comparison** | Multi-run comparison with statistical significance testing |
| **Runs** | Run browser, config comparison, export |
| **Config** | Run configuration inspection |
| **Monitoring** | Resource usage and bottleneck detection |
| **Card Meta** | Card registry breakdown and per-run card analysis |

### Advanced Metrics

- **Growth rate**: First derivative of fitness curve
- **Acceleration**: Second derivative of fitness curve
- **Convergence detection**: Rolling variance threshold
- **Pareto front**: Non-dominated solutions (fitness vs diversity)
- **Statistical significance**: Welch's t-test, Cohen's d effect size
- **Quantiles**: 25th, 50th, 75th, 90th, 95th percentiles

### Report Generation

```bash
# HTML report with charts
crp report --run-dir runs/run_123 --type training --format html

# Markdown report
crp report --run-dir runs/run_123 --type training --format markdown

# JSON metrics export
crp report --run-dir runs/run_123 --type training --format json

# Tournament report
crp report --run-dir runs/run_123 --type tournament --format html

# Compare two runs
crp compare --runs run_123 run_456 --format html
```

---

## ⚙️ Configuration System

### Schema-Based Validation

All configs are validated against schemas before use:

```python
from src.config import ConfigurationManager

config_manager = ConfigurationManager("configs/evolution.yaml")
validated_config = config_manager.validate()

# Get built-in templates
evolution_config = config_manager.get_template("evolution")
tournament_config = config_manager.get_template("tournament")
export_config = config_manager.get_template("export")
```

### Configuration Inheritance

Configs can inherit from other configs:

```yaml
# base_evolution.yaml
population:
  size: 200
  elite_count: 10

# custom_evolution.yaml
_base: base_evolution.yaml
population:
  size: 300  # Override size
mutation:
  strategy: "adaptive"  # Override strategy
```

### Environment Variable Substitution

```yaml
# Configs support ${VAR} substitution
training:
  max_generations: ${MAX_GENERATIONS:-500}  # Default 500
  runs_dir: ${RUNS_DIR:~/cr-pipeline/runs}
```

---

## 🧪 Data Augmentation

Augmentation strategies for training diversity:

| Type | Description | Intensity |
|------|-------------|-----------|
| **Deck Composition** | Swap cards in opponent decks | Light/Medium/Heavy |
| **Card Order** | Shuffle initial card order | Light/Medium/Heavy |
| **Opponent Strategy** | Vary aggression, timing, target priority | Light/Medium/Heavy |
| **Game Conditions** | Duration, elixir rate, overtime thresholds | Light/Medium/Heavy |
| **Elixir Advantage** | Start with elixir imbalance | Light/Medium/Heavy |
| **Timing** | Randomize deployment timing | Light/Medium/Heavy |

```python
from src.data import AugmentationConfig, AugmentationPipeline

config = AugmentationConfig(
    enabled=True,
    intensity=0.5,
    seed=42,
)
pipeline = AugmentationPipeline(config)
augmented_deck = pipeline.augment_deck(original_deck)
```

---

## 📁 Output Structure

```
runs/
├── run_<timestamp>_<name>/          # Per-run directory
│   ├── metrics.json                 # Training metrics
│   ├── fitness_history.json         # Per-generation fitness stats
│   ├── config.yaml                  # Run configuration snapshot
│   ├── checkpoints/                 # Checkpoint snapshots
│   │   ├── gen_0010/
│   │   │   ├── population.pt
│   │   │   ├── fitness_history.json
│   │   │   └── metadata.json
│   │   └── gen_0020/
│   ├── best_agent.pt                # Best agent weights
│   ├── best_agent_metadata.json     # Best agent metadata
│   ├── tournament_results.json      # Tournament evaluation results
│   └── elo_history.json             # ELO rating progression
├── run_<timestamp>_<name>2/        # Another run
└── best/                            # Continuously updated best agent
    ├── best_agent.pt                # Evolved policy genome
    └── metadata.json

reports/
├── training_report.html
├── tournament_report.html
└── comparison_report.html

models/
├── best_agent.onnx
├── best_agent.torchscript.pt
├── best_agent.npy
└── best_agent.json
```

---

### Agent checkpoint format

`best_agent.pt` is a torch-saved dict. The evolved policy genome is the trained
artefact, so it is what gets persisted:

| Key | Meaning |
|---|---|
| `genome` | The evolved policy vector (`PolicySpec.num_params` floats) |
| `param_kind` | `"genome"` or `"network"` — which parameter set is authoritative |
| `weights` | Same as `genome` when one is present; kept for older readers |
| `network_weights` | Torch parameters, only if a network was ever built |

Load with `EvolutionaryAgent.load_checkpoint`, or read the genome directly:

```python
import torch
genome = torch.load("runs/best/best_agent.pt", weights_only=False)["genome"]
```

The Torch network is built lazily and is *not* what training optimises, so a
checkpoint holding only network parameters cannot be played by the simulator —
`scripts/evaluate.py` rejects those explicitly rather than returning an
unusable vector.

---

## 🧠 Neural Architecture Search

Automatically discover optimal network topologies:

```bash
# Run architecture search
crp search --generations 20 --population 20 --budget 1000

# Search with specific constraints
crp search --generations 30 --population 30 \
  --max-params 1000000 --min-params 10000 \
  --preferred-arch cnn_lstm
```

Search space includes:
- **Layer types**: Conv2D, Conv1D, LSTM, GRU, Dense, BatchNorm, Attention, Residual
- **Filter sizes**: 16-256 channels
- **Layer counts**: 1-8 layers
- **Activation functions**: ReLU, LeakyReLU, Tanh, SiLU
- **Pooling**: MaxPool, AvgPool
- **Dropout**: 0.0-0.5
- **Residual connections**: Yes/No
- **Attention heads**: 1-8

---

## 📊 Model Ensembling

Combine top-performing models for improved robustness:

```python
from src.models import EnsembleBuilder, EnsembleMethod

builder = EnsembleBuilder()
ensemble = builder.build(
    model_paths=["runs/run_1/best_agent.pt", "runs/run_2/best_agent.pt"],
    method=EnsembleMethod.WEIGHT_AVERAGING,
    tournament_weights=True,
)
ensemble.save("models/ensemble_best.pt")
```

Methods:
- **Weight Averaging** (performance-weighted)
- **Geometric Mean** (diversity-preserving)
- **Stacking** (meta-learner)
- **Voting** (discrete action voting)

---

## 🕹️ Simulation Model

The simulator is deliberately low-fidelity but internally consistent.

**Timebase.** `TICKS_PER_SECOND = 10`, so regulation (1800 ticks) is 180
seconds of game time. Card stats are authored in real units — `attack_speed` in
seconds, `move_speed` in tiles per second — and converted through that
constant. Elixir defaults to the real single-elixir rate of 1 per 2.8s.

**Arena (8 columns x 6 rows).**

```
row 0   opponent king                    (col 3.5)      <- deepest
row 1   opponent princess towers         (cols 2, 5)
row 2   river / bridges                  (cols 3, 4)
row 3   mid-field
row 4   player princess towers           (cols 2, 5)
row 5   player king                      (col 3.5)      <- deepest
```

Princess towers sit closer to the river than the king, so attackers meet a
princess first. Ground troops must path to a bridge column to cross; air units
fly straight.

**King activation.** A king tower starts inactive and does not fire until it
takes damage or its side loses a princess tower.

**Scoring.** Crowns accrue as towers fall: 1 per princess tower, 3 for the
king. Destroying a king ends the match immediately. If crowns are level when
regulation expires the match goes to overtime; a crown lead ends it there.

**Deployment.** Troops may only be placed on their owner's half. Spells target
the entire arena.

**Card cycle.** The deck is a fixed rotation of 8 cards with 4 in hand. Playing
a card sends it to the back of the queue and draws the front card into that
slot; the other three slots are untouched.

---

## ⚠️ Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| **Sparse rewards** | Shaped rewards: tower damage, elixir efficiency, unit survival |
| **Non-stationary opponent** | Self-play; tournament evaluation; population-based training |
| **Large action space** | Hierarchical action: select card first, then placement; action masking |
| **Training slow on CPU** | Parallel simulation with multiprocessing; vectorized state |
| **Live-game latency** | Downsample frames; GPU inference; async action dispatch |
| **Card interaction complexity** | Start with single-deck evolution; gradually add deck diversity |
| **Overfitting to meta** | Data augmentation; diversity preservation; novelty search |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_sim_engine.py -v
pytest tests/test_evolution.py -v
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test coverage: 453 tests** across simulation engine, evolution strategies,
tournament system, checkpoint/resume, run artifacts, desktop UI,
visualization, and integration.

Several suites are worth calling out because they guard against silent failure
rather than crashes — the failure mode where everything is green and nothing is
learned:

- **`tests/test_tournament_training.py`** — Swiss pairing covers the field,
  ELO reflects results, and — the one that matters — the trained champion beats
  the generation-0 champion head to head. Relative fitness can rise without
  anything getting stronger; this catches that.
- **`tests/test_training_signal.py`** — fitness varies with the genome, is
  reproducible for a given seed, `evaluate_population` returns real scores, and
  selection raises mean fitness across generations.
- **`tests/test_ga_operators.py`** — every selection operator prefers fitter
  individuals regardless of population ordering, parent re-draws terminate, all
  mutation operators share one signature, and a seeded run reproduces.
- **`tests/test_persistence.py`** — a trained agent survives save/load and
  plays identically afterwards. Saving the wrong parameter set loses a training
  run without raising anything.
- **`tests/test_resume.py`** — continuing a run really does carry the
  population, generation counter, hall of fame and ratings, and seeding keeps
  the chosen agents intact.
- **`tests/test_run_artifacts.py`** — what a finished run leaves on disk: a
  small agent checkpoint, a checkpoint you can actually resume from, a
  non-empty log, and metrics that describe the run that happened. Every case
  came from inspecting real runs, not from reading code.
- **`tests/test_opponents.py`** — baselines play legally, respond to pushes,
  and remain in a band where untrained agents neither dominate nor are shut
  out.
- **`tests/test_sim_regressions.py`** — simulation behaviour that was
  previously wrong in ways no test detected (hand rotation, deployment zones,
  status effects, crown awards, death splits, attack cadence).

---

## 🔮 Roadmap

- [x] Phase 1: Simulation engine MVP (single arena, basic cards)
- [x] Phase 2: Parallel training infrastructure
- [x] Phase 3: Genetic algorithm core
- [x] Phase 4: Live visualization dashboard
- [x] Phase 5: Self-play training pipeline
- [x] Phase 6: Expanded card registry (30+ cards)
- [x] Phase 7: Curriculum learning with phase transitions
- [x] Phase 8: Tournament evaluation system
- [x] Phase 9: Neural architecture search
- [x] Phase 10: Model ensembling
- [x] Phase 11: Hyperparameter optimization
- [x] Phase 12: Experiment tracking
- [x] Phase 13: Pipeline orchestration
- [x] Phase 14: Data augmentation
- [x] Phase 15: Configuration validation system
- [x] Phase 16: CLI tool (11 commands)
- [x] Phase 17: Integration tests
- [x] Phase 18: Docker containerization (`Dockerfile`)
- [ ] Phase 19: Live-game interaction prototype
- [ ] Phase 20: Full simulation (all cards, arenas, mechanics)
- [ ] Phase 21: Live-game fine-tuning
- [ ] Phase 22: Distributed training (Ray)
- [ ] Phase 23: Auto-generated API documentation
- [ ] Phase 24: Jupyter tutorials

---

## 📄 License

[To be determined]

---

## 🙏 Acknowledgments

- [NEAT-Python](https://github.com/ntasfi/NEAT-Python) -- NeuroEvolution reference
- [Ray](https://docs.ray.io/) -- Distributed parallel training
- [Clash Royale Wiki](https://clashroyale.fandom.com/) -- Card data & mechanics reference
- [Optuna](https://optuna.org/) -- Hyperparameter optimization patterns
