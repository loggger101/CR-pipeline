# CR-Pipeline: Evolutionary Neural Network for Clash Royale

> **Automated AI agents learn to play Clash Royale through genetic evolution — from raw screen pixels to strategic gameplay.**

---

## 🎯 Project Overview

CR-Pipeline is a complete framework for training autonomous Clash Royale agents using **evolutionary algorithms** and **neural networks**. Instead of traditional reinforcement learning, we use a genetic approach:

1. **Population**: Maintain a population of neural network "brains"
2. **Evaluation**: Each brain plays matches (simulated or live)
3. **Selection**: Keep the top performers
4. **Mutation/Crossover**: Evolve the next generation
5. **Repeat**: Over hundreds of generations, agents develop sophisticated strategies

---

## 🏗️ Architecture

```
CR-Pipeline/
├── README.md
│
├── 1_game_interaction/          # Stage 1: Live game interaction
│   ├── screen_capture.py        # Screen capture (PyAutoGUI / mss)
│   ├── game_state_extractor.py  # Parse game state from screen
│   ├── action_engine.py         # Send inputs (mouse/keyboard / ADB)
│   └── live_match_runner.py     # Orchestrates live gameplay
│
├── 2_simulation_engine/          # Stage 2: Low-quality simulator
│   ├── simulator.py             # Core simulation loop
│   ├── physics.py               # Simplified game physics
│   ├── card_system.py           # Card deck, elixir, spawn logic
│   ├── pathfinding.py           # Unit movement on simplified map
│   ├── state_renderer.py        # Render low-quality frames for NN input
│   └── batch_simulator.py       # Run thousands of parallel sims
│
├── 3_neural_network/             # Stage 3: Neural network models
│   ├── brain.py                 # Base neural network class
│   ├── cnn_brain.py             # CNN-based vision brain (pixel input)
│   ├── rnn_brain.py             # RNN/LSTM brain (temporal sequences)
│   ├── hybrid_brain.py          # CNN + RNN hybrid (recommended)
│   ├── action_head.py           # Multi-output action head
│   └── model_zoo/               # Pre-built model architectures
│
├── 4_evolution/                  # Stage 4: Genetic algorithm
│   ├── fitness_evaluator.py     # Score matches, compute fitness
│   ├── selection.py             # Tournament / rank-based selection
│   ├── crossover.py             # Gene crossover operators
│   ├── mutation.py              # Mutation operators (Gaussian, swap, etc.)
│   ├── population.py            # Population management
│   ├── generation_manager.py    # Track generations, checkpoints
│   └── neuroevolution.py        # Main NEAT/ES evolution loop
│
├── 5_visualization/              # Stage 5: Live visualization
│   ├── dashboard.py             # Web dashboard (Flask/FastAPI + HTML)
│   ├── match_viewer.py          # Replay viewer
│   ├── fitness_tracker.py       # Live fitness curve plotting
│   ├── heatmap_overlay.py       # Action heatmap on game frames
│   └── stats_exporter.py        # Export stats to CSV/JSON
│
├── 6_training_pipeline/          # Stage 6: Orchestration
│   ├── train.py                 # Main training entry point
│   ├── config.yaml              # Hyperparameters, paths, settings
│   ├── distributed.py           # Multi-GPU / multi-machine support
│   ├── checkpoint.py            # Save/load checkpoints
│   └── sweep.py                 # Hyperparameter sweep runner
│
├── 7_assets/                     # Stage 7: Static assets
│   ├── maps/                    # Simplified Clash Royale maps
│   ├── card_sprites/            # Card icon templates
│   ├── fonts/                   # UI fonts for rendering
│   └── sounds/                  # Game sounds for simulation
│
├── 8_tests/                     # Testing
│   ├── test_simulation.py
│   ├── test_neural_network.py
│   ├── test_evolution.py
│   └── test_integration.py
│
├── requirements.txt
└── setup.py
```

---

## 📋 Stage-by-Stage Framework

### Stage 1: Game Interaction Module

**Goal**: Capture the game state and send actions to Clash Royale.

#### 1a. Screen Capture
```python
# screen_capture.py
import mss
import numpy as np

class ScreenCapture:
    """Capture the game window in real-time."""
    
    def __init__(self, region=None, fps=30):
        self.sct = mss.mss()
        self.region = region or self._get_game_window()
        self.fps = fps
        self.interval = 1000 / fps  # ms between frames
    
    def _get_game_window(self):
        """Auto-detect the Clash Royale window region."""
        # Use pygetwindow to find the game window
        ...
    
    def capture(self) -> np.ndarray:
        """Capture a single frame as a numpy array (BGR)."""
        screenshot = self.sct.grab(self.region)
        return np.array(screenshot)[:, :, :3]
    
    def capture_batch(self, n: int) -> list:
        """Capture n frames for batch processing."""
        return [self.capture() for _ in range(n)]
```

#### 1b. Game State Extraction
```python
# game_state_extractor.py
class GameStateExtractor:
    """Extract structured game state from raw screen frames."""
    
    def __init__(self):
        self.card_detector = CardDetectionModel()
        self.health_estimator = HealthEstimationModel()
        self.tower_detector = TowerDetectionModel()
    
    def extract(self, frame: np.ndarray) -> GameState:
        """
        Parse a game frame into a structured GameState object.
        
        GameState includes:
        - player_arena (ArenaPosition)
        - opponent_arena (ArenaPosition)
        - player_elixir (float, 0-10)
        - opponent_elixir (float, 0-10)
        - player_deck (list[Card], current hand)
        - player_king_health (float)
        - opponent_king_health (float)
        - player_princess_healths (list[float])
        - opponent_princess_healths (list[float])
        - units_on_field (list[Unit])
        - arena_layout (ArenaMap)
        - timer (float, seconds remaining)
        """
        ...
    
    class GameState:
        """Structured representation of the game state."""
        player_elixir: float
        opponent_elixir: float
        player_deck: list[Card]
        player_king_health: float
        opponent_king_health: float
        player_princess_healths: list[float]
        opponent_princess_healths: list[float]
        units_on_field: list[Unit]
        arena_layout: ArenaMap
        timer: float
```

#### 1c. Action Engine
```python
# action_engine.py
class ActionEngine:
    """Send game actions (card play, target placement)."""
    
    def __init__(self, mode="auto"):
        """
        mode: 'auto' (mouse/keyboard), 'adb' (Android), 'memory' (direct)
        """
        ...
    
    def play_card(self, card_index: int, target_x: float, target_y: float):
        """
        Play a card from the deck at the specified target location.
        
        Args:
            card_index: Index of the card in the current deck (0-3)
            target_x: Normalized X coordinate (0-1) on the arena
            target_y: Normalized Y coordinate (0-1) on the arena
        """
        ...
    
    def select_card(self, card_index: int):
        """Select a card from the deck (without placing it yet)."""
        ...
    
    def release_card(self, target_x: float, target_y: float):
        """Release the selected card at the target location."""
        ...
```

#### 1d. Live Match Runner
```python
# live_match_runner.py
class LiveMatchRunner:
    """Run a single live match with a given neural network brain."""
    
    def __init__(self, brain: Brain, capture: ScreenCapture, engine: ActionEngine):
        self.brain = brain
        self.capture = capture
        self.engine = engine
        self.episode_reward = 0
        self.steps = 0
    
    def run_episode(self, max_steps: int = 600) -> MatchResult:
        """Run one complete match (up to max_steps actions)."""
        for step in range(max_steps):
            frame = self.capture.capture()
            state = self.extract_state(frame)
            
            # Brain decides action
            action = self.brain.act(state)
            
            # Execute action in game
            self.engine.execute(action)
            
            # Compute reward
            reward = self.compute_reward(state)
            self.episode_reward += reward
            self.steps += 1
        
        return MatchResult(
            reward=self.episode_reward,
            steps=self.steps,
            winner="player" if self.won else "opponent",
            frames=self.capture.frames
        )
```

---

### Stage 2: Simulation Engine

**Goal**: Create a fast, low-fidelity simulator for mass parallel training.

#### 2a. Core Simulator
```python
# simulator.py
class Simulator:
    """
    Low-quality Clash Royale simulator for mass parallel training.
    
    Trade fidelity for speed: simplified map, approximate physics,
    abstracted card effects. Designed to run 1000s of instances.
    """
    
    def __init__(self, config: SimConfig):
        self.config = config
        self.arena = SimplifiedArena(config.arena_type)
        self.unit_factory = UnitFactory(config.unit_types)
        self.card_system = CardSystem(config.card_definitions)
        self.fps = config.sim_fps  # e.g., 60 for smooth sim
        self.step_time = 1.0 / self.fps
    
    def reset(self, deck1: list[Card], deck2: list[Card]) -> GameState:
        """Reset the simulation with two player decks."""
        ...
    
    def step(self, action1: Action, action2: Action) -> SimResult:
        """
        Advance the simulation by one step.
        
        Both players act simultaneously (with timing resolution).
        Returns the result of this step.
        """
        # Process card plays
        self._process_card_plays(action1, action2)
        
        # Update unit positions
        self._update_units()
        
        # Process combat
        self._process_combat()
        
        # Check win conditions
        result = self._check_game_state()
        
        return SimResult(
            state=self.get_state(),
            reward1=self._compute_reward(action1),
            reward2=self._compute_reward(action2),
            done=result.done,
            winner=result.winner
        )
    
    def render_frame(self, state: GameState) -> np.ndarray:
        """Render a low-quality frame for neural network input."""
        # Simple pixel art rendering
        ...
```

#### 2b. Simplified Arena Map
```python
# The arena is represented as a 2D grid or simplified polygon map:
#
#   Opponent King Tower (center)
#   Opponent Princess Tower (left)    Opponent Princess Tower (right)
#   Opponent Bridge (left)            Opponent Bridge (right)
#   River
#   Player Bridge (left)              Player Bridge (right)
#   Player Princess Tower (left)      Player Princess Tower (right)
#   Player King Tower (center)
#
# Units move along paths toward their targets.
# Cards spawn at bridge positions or designated spawn points.
```

#### 2c. Batch Simulator
```python
# batch_simulator.py
class BatchSimulator:
    """Run thousands of simulator instances in parallel."""
    
    def __init__(self, num_instances: int = 1000):
        self.num_instances = num_instances
        self.simulators = [Simulator() for _ in range(num_instances)]
        self.batch_states = None
        self.batch_rewards = None
    
    def reset_batch(self, decks: list[list[Card]]) -> np.ndarray:
        """Reset all instances and return initial frames."""
        ...
    
    def step_batch(self, actions: np.ndarray) -> BatchResult:
        """
        Step all instances with vectorized operations.
        
        Args:
            actions: np.ndarray of shape (num_instances, action_dim)
        
        Returns:
            BatchResult with frames, rewards, dones, infos
        """
        # Vectorized physics updates
        ...
        return BatchResult(
            frames=rendered_frames,      # (N, H, W, C)
            rewards=rewards,              # (N,)
            dones=dones,                  # (N,) boolean
            infos=infos                   # metadata per instance
        )
```

---

### Stage 3: Neural Network Models

**Goal**: Neural network architectures that map game states to actions.

#### 3a. Base Brain Class
```python
# brain.py
from abc import ABC, abstractmethod
import numpy as np

class Brain(ABC):
    """Abstract base class for neural network brains."""
    
    def __init__(self, action_space: ActionSpace):
        self.action_space = action_space
        self.model = None
        self.is_training = False
    
    @abstractmethod
    def act(self, state: GameState) -> Action:
        """Choose an action given a game state."""
        ...
    
    @abstractmethod
    def observe(self, transition: Transition):
        """Store a transition for training."""
        ...
    
    @abstractmethod
    def train(self):
        """Update weights from stored transitions."""
        ...
    
    @abstractmethod
    def save(self, path: str):
        """Save model weights."""
        ...
    
    @abstractmethod
    def load(self, path: str):
        """Load model weights."""
        ...
    
    @property
    @abstractmethod
    def gene(self) -> dict:
        """Return the brain's weights as a flat vector for evolution."""
        ...
    
    @gene.setter
    @abstractmethod
    def gene(self, vector: np.ndarray):
        """Set the brain's weights from a flat vector."""
        ...
```

#### 3b. Hybrid CNN+RNN Brain (Recommended)
```python
# hybrid_brain.py
import torch
import torch.nn as nn

class HybridBrain(Brain):
    """
    CNN + RNN brain for Clash Royale.
    
    Architecture:
    1. CNN encoder processes frames (or state features)
    2. RNN/LSTM maintains temporal memory
    3. Multiple action heads output different action types
    """
    
    def __init__(self, action_space: ActionSpace, config: BrainConfig):
        super().__init__(action_space)
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),   # Frame input
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # RNN temporal processing
        self.rnn = nn.LSTM(
            input_size=config.cnn_output_dim,
            hidden_size=config.hidden_size,
            num_layers=config.rnn_layers,
            batch_first=True
        )
        
        # Action heads
        self.card_select_head = nn.Linear(config.hidden_size, 4)  # Select card
        self.target_head = nn.Linear(config.hidden_size, 2)        # Target coords
        self.timing_head = nn.Linear(config.hidden_size, 1)        # Timing delay
        self.aggression_head = nn.Linear(config.hidden_size, 1)    # Aggression level
        
        self.config = config
        self.hidden_state = None
    
    def act(self, state: GameState) -> Action:
        """Forward pass to produce an action."""
        frame = self._render_state(state)  # (H, W, 3)
        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)
        
        features = self.cnn(frame)
        features = features.view(1, 1, -1)
        
        if self.hidden_state is None:
            self.hidden_state = (
                torch.zeros(1, 1, self.config.hidden_size),
                torch.zeros(1, 1, self.config.hidden_size)
            )
        
        output, self.hidden_state = self.rnn(features, self.hidden_state)
        output = output.squeeze(1)
        
        # Decode actions
        card_idx = torch.argmax(self.card_select_head(output)).item()
        target_x = torch.sigmoid(self.target_head(output)[0]).item()
        target_y = torch.sigmoid(self.target_head(output)[1]).item()
        timing = torch.sigmoid(self.timing_head(output)).item()
        aggression = torch.sigmoid(self.aggression_head(output)).item()
        
        return Action(
            card_index=card_idx,
            target_x=target_x,
            target_y=target_y,
            timing=timing,
            aggression=aggression
        )
    
    @property
    def gene(self) -> np.ndarray:
        """Flatten all weights into a single vector."""
        weights = []
        for param in self.parameters():
            weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(weights)
    
    @gene.setter
    def gene(self, vector: np.ndarray):
        """Set all weights from a flat vector."""
        offset = 0
        for param in self.parameters():
            size = param.numel()
            param.data = torch.tensor(
                vector[offset:offset + size].reshape(param.shape)
            )
            offset += size
```

#### 3c. Action Space Definition
```python
# action_head.py
@dataclass
class ActionSpace:
    """Defines the possible actions the brain can take."""
    
    # Card selection (4 cards in hand)
    card_options: int = 4
    
    # Target position (normalized arena coordinates)
    target_x_range: tuple = (0.0, 1.0)
    target_y_range: tuple = (0.0, 1.0)
    
    # Timing (when to release the card)
    timing_bins: int = 10
    
    # Aggression level
    aggression_bins: int = 5
    
    @property
    def action_dim(self) -> int:
        """Continuous action dimension for the network output."""
        return 4  # card_idx (discrete), target_x, target_y, timing
    
    def sample_random(self) -> Action:
        """Sample a random valid action."""
        return Action(
            card_index=random.randint(0, self.card_options - 1),
            target_x=random.uniform(*self.target_x_range),
            target_y=random.uniform(*self.target_y_range),
            timing=random.uniform(0, 1),
            aggression=random.uniform(0, 1)
        )
```

---

### Stage 4: Evolution Engine

**Goal**: Genetic algorithm that evolves brain populations over generations.

#### 4a. Fitness Evaluator
```python
# fitness_evaluator.py
class FitnessEvaluator:
    """Compute fitness scores for evolved brains."""
    
    def __init__(self, config: FitnessConfig):
        self.config = config
    
    def evaluate(self, brain: Brain, num_episodes: int = 10) -> float:
        """
        Evaluate a brain by playing multiple matches.
        
        Fitness = weighted sum of:
        - Match result (win/loss/draw)
        - King tower damage dealt
        - King tower damage taken (inverted)
        - Princess tower damage dealt
        - Elixir efficiency (damage per elixir)
        - Average elixir per turn
        - Consistency (std dev of rewards across episodes)
        """
        rewards = []
        for _ in range(num_episodes):
            result = self._play_match(brain)
            fitness = self._compute_fitness(result)
            rewards.append(fitness)
        
        # Use mean fitness with small penalty for variance
        return np.mean(rewards) - 0.1 * np.std(rewards)
    
    def _compute_fitness(self, result: MatchResult) -> float:
        """Compute fitness from a single match result."""
        score = 0.0
        
        # Match result (primary)
        if result.winner == "player":
            score += 100.0
        elif result.winner == "draw":
            score += 50.0
        
        # Tower damage
        score += result.king_tower_damage * 2.0
        score += result.princess_tower_damage * 0.5
        
        # Elixir efficiency
        score += result.damage_per_elixir * 5.0
        
        # Time bonus (faster wins are better)
        score += (360 - result.match_duration) * 0.1
        
        return score
```

#### 4b. Selection
```python
# selection.py
class SelectionStrategy(ABC):
    """Abstract selection strategy."""
    @abstractmethod
    def select(self, population: list[Brain], fitnesses: np.ndarray, k: int) -> list[Brain]:
        ...

class TournamentSelection(SelectionStrategy):
    """Select the best from a random tournament."""
    
    def __init__(self, tournament_size: int = 5):
        self.tournament_size = tournament_size
    
    def select(self, population, fitnesses, k):
        selected = []
        for _ in range(k):
            indices = np.random.choice(len(population), self.tournament_size, replace=False)
            winner_idx = indices[np.argmax(fitnesses[indices])]
            selected.append(population[winner_idx])
        return selected

class RankBasedSelection(SelectionStrategy):
    """Selection probability proportional to rank."""
    
    def select(self, population, fitnesses, k):
        ranks = np.argsort(np.argsort(fitnesses))  # Higher rank = better
        probs = ranks / ranks.sum()
        indices = np.random.choice(len(population), k, p=probs)
        return [population[i] for i in indices]
```

#### 4c. Mutation
```python
# mutation.py
class MutationOperator(ABC):
    """Abstract mutation operator."""
    @abstractmethod
    def mutate(self, gene: np.ndarray) -> np.ndarray:
        ...

class GaussianMutation(MutationOperator):
    """Add Gaussian noise to weights."""
    
    def __init__(self, sigma: float = 0.01, rate: float = 0.1):
        self.sigma = sigma
        self.rate = rate  # Probability of mutating each weight
    
    def mutate(self, gene):
        mask = np.random.random(gene.shape) < self.rate
        noise = np.random.normal(0, self.sigma, gene.shape)
        return gene + noise * mask

class SwapMutation(MutationOperator):
    """Swap random subsets of weights."""
    
    def __init__(self, swap_rate: float = 0.05):
        self.swap_rate = swap_rate
    
    def mutate(self, gene):
        gene = gene.copy()
        n = len(gene)
        num_swaps = int(n * self.swap_rate)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            gene[i], gene[j] = gene[j], gene[i]
        return gene
```

#### 4d. Crossover
```python
# crossover.py
class CrossoverOperator(ABC):
    """Abstract crossover operator."""
    @abstractmethod
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        ...

class SinglePointCrossover(CrossoverOperator):
    """Single-point crossover."""
    
    def crossover(self, parent1, parent2):
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

class BlendCrossover(CrossoverOperator):
    """Blend between parents (BLX-α)."""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def crossover(self, parent1, parent2):
        low = np.minimum(parent1, parent2)
        high = np.maximum(parent1, parent2)
        spread = high - low
        child1 = np.random.uniform(low, high + self.alpha * spread)
        child2 = np.random.uniform(low, high - self.alpha * spread)
        return child1, child2
```

#### 4e. Population Manager
```python
# population.py
class Population:
    """Manage the evolving population of brains."""
    
    def __init__(self, size: int, action_space: ActionSpace, brain_class: type):
        self.size = size
        self.brain_class = brain_class
        self.action_space = action_space
        self.bra