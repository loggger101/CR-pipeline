# CR-Pipeline: Evolutionary Neural Network for Clash Royale

> An end-to-end pipeline for training Clash Royale AI agents using genetic/evolutionary algorithms — from live-game interaction to parallel simulation training and live visualization.

---

## 🎯 Goal

Train a population of neural networks that can play Clash Royale competitively using evolutionary strategies. The pipeline supports:

- **Live-game interaction** — a model plays the real game by interpreting the screen and issuing commands.
- **Low-fidelity simulation** — a simplified game engine allows thousands of parallel training instances.
- **Live visualization** — real-time dashboards showing training progress, fitness distributions, and agent behavior.

---

## 🏗️ Project Architecture

```
CR-Pipeline/
├── src/
│   ├── env/                    # Game environments
│   │   ├── live/               # Live-game interaction
│   │   │   ├── screen_capture.py   # Screen capture & preprocessing
│   │   │   ├── game_state.py       # Game state extraction from frames
│   │   │   ├── action_mapper.py    # Neural net output → in-game actions
│   │   │   └── overlay.py          # Optional: draw overlay on game window
│   │   └── sim/                # Low-fidelity simulation engine
│   │       ├── engine.py           # Core game loop (simplified rules)
│   │       ├── entities.py         # Cards, towers, elixir, etc.
│   │       ├── actions.py          # Discrete action space
│   │       ├── rendering.py        # Lightweight visual output
│   │       └── parallel_runner.py  # Run N instances concurrently
│   ├── models/                 # Neural network definitions
│   │   ├── agent.py            # Base agent class (forward pass, action selection)
│   │   ├── architecture.py     # Network architecture (CNN + LSTM / Transformer)
│   │   ├── population.py       # Population management (genotypes, fitness)
│   │   └── evolution.py        # Genetic algorithm: selection, crossover, mutation
│   ├── train/                  # Training loop & management
│   │   ├── trainer.py          # Main evolution loop
│   │   ├── evaluator.py        # Fitness evaluation (wins, trophies, etc.)
│   │   ├── checkpoint.py       # Save/load populations
│   │   └── hyperparams.py      # Configurable GA parameters
│   └── viz/                    # Live visualization
│       ├── dashboard.py        # Streamlit / Gradio dashboard
│       ├── metrics.py          # Fitness curves, stats
│       ├── replay.py           # Replay viewer for trained agents
│       └── live_game_view.py   # Overlay view during live gameplay
├── configs/
│   ├── live_game.yaml          # Live-game config (screen coords, hotkeys)
│   ├── sim_game.yaml           # Simulation config (map, balance)
│   └── evolution.yaml          # GA hyperparameters
├── assets/
│   ├── card_data.json          # Card stats, costs, behaviors
│   └── maps/                   # Map layouts for simulation
├── tests/
│   ├── test_sim_engine.py
│   ├── test_evolution.py
│   └── test_agent.py
├── scripts/
│   ├── train_sim.py            # Launch simulation training
│   ├── play_live.py            # Launch live-game agent
│   ├── launch_dashboard.py     # Start visualization dashboard
│   └── evaluate.py             # Evaluate top agents
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 📦 Components in Detail

### 1. Live-Game Interaction (`src/env/live/`)

| File | Purpose |
|------|---------|
| `screen_capture.py` | Captures the Clash Royale game window in real-time (using `pyautogui`, `mss`, or `dxcam`). Processes frames to a fixed resolution (e.g., 256×256) for the neural net input. |
| `game_state.py` | Extracts game state from frames: card hand, elixir levels, unit positions, tower health, arena. Uses template matching, color detection, or a lightweight CNN classifier. |
| `action_mapper.py` | Maps the neural network's output (continuous or discrete) to in-game actions: select card, place on target cell, deploy direction, etc. Includes debouncing, cooldown validation, and hotkey simulation. |
| `overlay.py` | Optionally draws bounding boxes, predicted actions, and fitness info as an overlay on the game window for debugging. |

**Input pipeline:**
```
Game Window → Screen Capture → Preprocess (resize, normalize) → State Extraction
                                                                    ↓
                                               Neural Net → Action Selection → Action Mapper → In-Game Input
```

**Key considerations:**
- Frame rate: target ≥ 15 FPS for real-time play.
- Input tensor: multi-channel (card hand as one-hot channels, elixir levels, tower HP heatmaps, etc.).
- Action space: discrete (select card + grid cell) or hybrid (card index + continuous position).

---

### 2. Low-Fidelity Simulation (`src/env/sim/`)

A simplified Clash Royale game engine designed for **massively parallel training**. Trade visual fidelity for speed — each instance runs in a lightweight process or thread.

| File | Purpose |
|------|---------|
| `engine.py` | Core game loop: elixir regeneration, unit spawning, movement, combat, tower damage. Runs at ~60+ ticks/sec in pure Python or NumPy. |
| `entities.py` | Card definitions (HP, damage, speed, target preference, deployment zone), tower definitions, arena layout. |
| `actions.py` | Discrete action space: `PLAY_CARD(card_idx, x, y)` where `(x, y)` is a grid cell on the player's half of the arena. |
| `parallel_runner.py` | Manages N parallel instances using `multiprocessing` or `ray`. Handles fitness collection, checkpointing, and worker lifecycle. |

**Simulation design principles:**
- **Tick-based**: Game logic advances in discrete ticks (e.g., 0.5s per tick).
- **State representation**: Compact NumPy arrays or dictionaries per instance — no GUI.
- **Parallelism**: Each agent instance runs in its own process; fitness aggregated by the trainer.
- **Determinism**: Seed-based RNG for reproducibility of match outcomes.

**Fitness function:**
```python
def evaluate_fitness(match_result):
    return (
        0.4 * trophy_gain                    # Primary: trophies won
      + 0.3 * (towers_destroyed / 6)         # Secondary: progress
      + 0.2 * (opponent_towers_destroyed == 0)  # Win bonus
      + 0.1 * (match_duration / max_duration)  # Efficiency: faster wins = better
    )
```

---

### 3. Evolutionary Neural Network (`src/models/`)

| File | Purpose |
|------|---------|
| `architecture.py` | Neural network definition. Options: (a) CNN + LSTM for frame-based input, (b) Vision Transformer, or (c) two-stream (CNN for vision + MLP for state features). Built with PyTorch. |
| `agent.py` | Agent wrapper: `forward(state) → logits`, `select_action(logits, epsilon)`, `save/load weights`. Supports both exploration (epsilon-greedy, softmax) and greedy inference. |
| `population.py` | Population management: stores genotypes (network weights), fitness scores, rankings, and supports elite preservation. |
| `evolution.py` | Genetic algorithm core: (a) **Selection** — tournament or rank-based, (b) **Crossover** — blend or single-point weight blending, (c) **Mutation** — Gaussian noise on weights, (d) **Elitism** — top N preserved unchanged, (e) **NSGA-II** or simple NEAT-style if topology evolves. |

**Evolution loop:**
```
1. Initialize population of N agents (random weights)
2. For each generation:
   a. Spawn N simulation instances
   b. Run each agent for M matches in parallel
   c. Evaluate fitness for each agent (average over M matches)
   d. Select parents (tournament selection)
   e. Apply crossover + mutation to produce offspring
   f. Replace population (with elitism)
   g. Log metrics, checkpoint best agent
3. Repeat until convergence or budget exhausted
```

**Key hyperparameters** (`configs/evolution.yaml`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 200 | Number of agents per generation |
| `matches_per_agent` | 5 | Matches each agent plays per generation |
| `mutation_rate` | 0.05 | Probability per weight to mutate |
| `mutation_std` | 0.1 | Std dev of Gaussian mutation noise |
| `crossover_rate` | 0.7 | Probability of crossover vs copy |
| `elite_count` | 10 | Number of top agents preserved |
| `tournament_size` | 5 | Tournament selection size |
| `max_generations` | 500 | Training budget |

---

### 4. Training Management (`src/train/`)

| File | Purpose |
|------|---------|
| `trainer.py` | Orchestrates the evolution loop: launches parallel simulation, collects fitness, triggers evolution, handles checkpoints and early stopping. |
| `evaluator.py` | Runs evaluation matches (against fixed baselines or self-play). Supports tournament mode. |
| `checkpoint.py` | Saves/loads population weights, fitness history, and training metadata. Uses `.pt` or `.npz` formats. |
| `hyperparams.py` | Loads and validates YAML configs. Provides defaults. |

---

### 5. Live Visualization (`src/viz/`)

| File | Purpose |
|------|---------|
| `dashboard.py` | Streamlit/Gradio web app showing: (a) fitness curves over generations, (b) top agent live gameplay, (c) population statistics (mean/max/min fitness), (d) hyperparameter controls. |
| `metrics.py` | Computes rolling averages, percentiles, and Paretos of fitness distributions. |
| `replay.py` | Viewer for recorded simulation matches — step through, watch, analyze agent decisions. |
| `live_game_view.py` | Real-time view during live-game play: screen capture + neural net predictions + fitness overlay. |

**Dashboard layout:**
```
┌─────────────────────────────────────────────────────────┐
│  CR-Pipeline Training Dashboard                         │
├──────────────────┬──────────────────────────────────────┤
│  Fitness Curves  │  Population Stats                    │
│  (generation vs  │  Best:  ██████████  1420           │
│   fitness)       │  Mean:  ██████        980            │
│                  │  Min:   ████          620            │
│                  │  Gen: 42 / 500                       │
├──────────────────┴──────────────────────────────────────┤
│  Top Agent Live View  │  Card Play Heatmap              │
│  [simulation frame]   │  [action frequency chart]       │
├─────────────────────────────────────────────────────────┤
│  Controls: [Pause] [Resume] [Checkpoint] [Export]       │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended for training; CPU works for simulation)

### Installation
```bash
git clone https://github.com/loggger101/CR-pipeline.git
cd CR-pipeline
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Train agents in simulation
python scripts/train_sim.py --config configs/sim_game.yaml --evolution configs/evolution.yaml

# 2. Launch live visualization dashboard
python scripts/launch_dashboard.py

# 3. Evaluate top checkpoint
python scripts/evaluate.py --checkpoint runs/gen_100/checkpoint.pt --matches 20

# 4. Deploy best agent to live game
python scripts/play_live.py --checkpoint runs/gen_500/checkpoint.pt
```

---

## 📊 Training Phases

| Phase | Description | Duration | Goal |
|-------|-------------|----------|------|
| **Phase 1** | Random policy baseline in simulation | 10 gen | Verify pipeline works |
| **Phase 2** | Evolution in simplified arena | 50 gen | Learn basic card placement |
| **Phase 3** | Evolution in full simulation | 100 gen | Learn deck synergy & timing |
| **Phase 4** | Fine-tune on live game | 100 gen | Adapt to real-game noise |
| **Phase 5** | Self-play evolution | 200 gen | Push to competitive level |

---

## 🧬 Evolution Strategy Details

### Network Architecture Options

**Option A — Frame-based (CNN + LSTM):**
```
Input: [batch, frames, channels, H, W]
  → Conv2d (32 ch) → ReLU → MaxPool
  → Conv2d (64 ch) → ReLU → MaxPool
  → Flatten
  → LSTM (256 hidden, 2 layers)
  → Dense (128) → ReLU
  → Dense (action_dim)  [output: action logits]
```

**Option B — State-augmented (CNN + MLP):**
```
Vision branch:  Input frames → CNN → [128]
State branch:   [card hand, elixir, tower HP] → MLP → [128]
  → Concat [256] → Dense → Action logits
```

### Genetic Algorithm Variants

| Variant | When to Use |
|---------|-------------|
| **NES** (NeuroEvolution of Augmenting Topologies) | Evolving both weights and topology |
| **CMA-ES** | High-dimensional weight space, smooth fitness landscape |
| **Simple GA** (tournament + blend crossover + Gaussian mutation) | Baseline, easy to parallelize |
| **MAP-Elites** | Diverse behavior exploration |

---

## 🎮 Action Space Design

```
Action = (card_index, target_x, target_y)

card_index:  0–7  (8-card hand; -1 = no action)
target_x:    0–W  (grid cells on player's half)
target_y:    0–H  (grid cells on player's half)

Total discrete actions: 8 × W × H + 1  (pass)
```

For efficiency, the action can be decomposed:
1. **Card selection** (discrete, 8 choices + pass)
2. **Placement** (continuous position, clipped to valid deployment zone)

This reduces the action space and allows finer-grained targeting.

---

## 📁 Output Structure

```
runs/
├── gen_001/
│   ├── population.pt
│   ├── fitness_history.json
│   └── metrics.json
├── gen_002/
│   ├── ...
├── best/
│   └── checkpoint.pt          # Continuously updated
└── checkpoints/
    └── gen_100/
        ├── population.pt
        └── fitness_history.json
```

---

## ⚠️ Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|------------|
| **Sparse rewards** (only at match end) | Shaped rewards: tower damage, elixir efficiency, unit survival |
| **Non-stationary opponent** | Self-play; fixed opponent baselines; population-based training |
| **Large action space** | Hierarchical action: select card first, then placement; action masking |
| **Training slow on CPU** | Ray for distributed parallelism; vectorized simulation with NumPy/JAX |
| **Live-game latency** | Downsample frames; run inference on GPU; async action dispatch |
| **Card interaction complexity** | Start with single-deck evolution; gradually add deck diversity |

---

## 📝 Roadmap

- [ ] Phase 1: Simulation engine MVP (single arena, basic cards)
- [ ] Phase 2: Parallel training infrastructure
- [ ] Phase 3: Genetic algorithm core (selection, crossover, mutation)
- [ ] Phase 4: Live visualization dashboard
- [ ] Phase 5: Live-game interaction prototype
- [ ] Phase 6: Full simulation (all cards, arenas, mechanics)
- [ ] Phase 7: Live-game fine-tuning
- [ ] Phase 8: Self-play competitive evolution
- [ ] Documentation & examples

---

## 📄 License

[To be determined]

---

## 🙏 Acknowledgments

- [NEAT-Python](https://github.com/ntasfi/NEAT-Python) — NeuroEvolution reference
- [Ray](https://docs.ray.io/) — Distributed parallel training
- [Clash Royale Wiki](https://clashroyale.fandom.com/) — Card data & mechanics reference
