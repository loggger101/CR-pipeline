"""Evolutionary agent wrapper.

Encapsulates a neural network with action selection, exploration,
and weight management capabilities. Supports both greedy inference
and epsilon-greedy exploration.

Enhanced with:
- Temperature-scaled softmax card selection
- Action masking for invalid cards
- Multiple exploration strategies (epsilon-greedy, Boltzmann, entropy)
- Better exploration rate management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

from .architecture import (
    AgentArchitecture, CNNLSTMAgent, CNNMLPAgent,
    create_cnn_lstm_agent, create_cnn_mlp_agent,
)

logger = logging.getLogger(__name__)


def _looks_like_genome(weights) -> bool:
    """Whether a stored parameter vector is a policy genome.

    Used only for checkpoints written before ``param_kind`` existed. The two
    representations differ by three orders of magnitude in length, so length
    is a reliable discriminator.
    """
    if weights is None:
        return False
    from .policy import DEFAULT_POLICY_SPEC
    return int(np.asarray(weights).size) == DEFAULT_POLICY_SPEC.num_params


class ExplorationStrategy:
    """Exploration strategy types."""
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    ENTROPY_REGULARIZED = "entropy_regularized"
    BOGOTA = "bogota"  # Bayesian Optimization-guided


@dataclass
class AgentConfig:
    """Configuration for an evolutionary agent.

    Attributes:
        architecture: Type of neural network architecture.
        input_channels: Number of input channels for the network.
        input_height: Height of input tensor.
        input_width: Width of input tensor.
        hidden_size: Size of hidden layers.
        lstm_layers: Number of LSTM layers (for CNN+LSTM).
        epsilon: Exploration rate (epsilon-greedy).
        epsilon_decay: Rate at which epsilon decays.
        epsilon_min: Minimum epsilon value.
        grid_cols: Grid columns for placement.
        grid_rows: Grid rows for placement.
        device: Device for inference.
        temperature: Temperature for softmax card selection.
        exploration_strategy: Type of exploration strategy.
        entropy_coeff: Coefficient for entropy regularization.
    """
    architecture: str = "cnn_lstm"
    input_channels: int = 12
    input_height: int = 64
    input_width: int = 64
    hidden_size: int = 256
    lstm_layers: int = 2
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    grid_cols: int = 8
    grid_rows: int = 6
    device: str = "cpu"
    temperature: float = 1.0
    exploration_strategy: str = ExplorationStrategy.EPSILON_GREEDY
    entropy_coeff: float = 0.0

    @property
    def arch(self) -> AgentArchitecture:
        """Get the AgentArchitecture config."""
        return AgentArchitecture(
            name=self.architecture,
            input_channels=self.input_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
        )


class EvolutionaryAgent:
    """Wrapper for a neural network agent in evolutionary training.

    Handles:
    - Forward pass / inference
    - Action selection with exploration (multiple strategies)
    - Weight management (get/set)
    - Checkpoint save/load
    - Exploration rate management

    The agent outputs:
    - Card selection logits: shape (5,) for 4 cards + pass
    - Placement coordinates: shape (2,) continuous [col, row] in [-1, 1]
    """

    def __init__(self, config: Optional[AgentConfig] = None,
                 weights: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """Initialize the agent.

        Args:
            config: Agent configuration.
            weights: Initial weights (if any).
            seed: Random seed.
        """
        self.config = config or AgentConfig()
        self.rng = np.random.RandomState(seed)
        self.device = torch.device(self.config.device)

        # The Torch network is built on first use. Constructing a CNN+LSTM per
        # agent up front made initialising a 200-agent population spend ~9s
        # (and hundreds of MB) building networks that the evolutionary path --
        # which evolves the compact policy genome in models/policy.py -- never
        # touches.
        self._network = None

        # Exploration
        self.epsilon = self.config.epsilon
        self.weights = weights
        self._prev_epsilon = self.epsilon

        # Statistics
        self.total_actions = 0
        self.exploration_actions = 0
        self.card_selection_counts: dict = {}
        self.position_heatmap: list = []

        # Track best weights for evolutionary context
        self.best_fitness: float = -float('inf')
        self.best_weights: Optional[np.ndarray] = None

    @property
    def network(self):
        """The Torch network, constructed on first access."""
        if self._network is None:
            if self.config.architecture == "cnn_lstm":
                self._network = create_cnn_lstm_agent(
                    input_channels=self.config.input_channels,
                    input_height=self.config.input_height,
                    input_width=self.config.input_width,
                    hidden_size=self.config.hidden_size,
                    lstm_layers=self.config.lstm_layers,
                    device=self.config.device,
                )
            else:
                self._network = create_cnn_mlp_agent(
                    input_channels=self.config.input_channels,
                    input_height=self.config.input_height,
                    input_width=self.config.input_width,
                    hidden_size=self.config.hidden_size,
                    device=self.config.device,
                )
        return self._network

    @property
    def has_network(self) -> bool:
        """Whether the Torch network has been built yet."""
        return self._network is not None

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the network forward pass.

        Args:
            state: Input state tensor of shape (C, H, W).

        Returns:
            card_logits: numpy array of shape (5,)
            placement: numpy array of shape (2,) in [-1, 1]
        """
        self.network.eval()
        with torch.no_grad():
            # Add batch dimension
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            card_logits, placement = self.network(x)

        return card_logits.squeeze(0).cpu().numpy(), placement.squeeze(0).cpu().numpy()

    def select_action(self, state: np.ndarray,
                      valid_cards: Optional[np.ndarray] = None,
                      valid_positions: Optional[np.ndarray] = None,
                      temperature: Optional[float] = None) -> dict:
        """Select an action with exploration.

        Args:
            state: Input state tensor.
            valid_cards: Boolean mask of valid card indices (4,).
                         True = card can be played.
            valid_positions: Boolean mask of valid positions (unused currently).
            temperature: Override temperature for card selection.

        Returns:
            Action dict with:
            - card_idx: Selected card index (-1 for pass)
            - target_col: Target column
            - target_row: Target row
            - is_exploration: Whether this was an exploratory action
            - card_logits: Raw logits for logging
            - card_probs: Softmax probabilities for logging
        """
        self.total_actions += 1

        card_logits, placement = self.forward(state)

        # Apply action masking to logits (prevent selecting invalid cards)
        if valid_cards is not None:
            card_logits = self._apply_action_mask(card_logits, valid_cards)

        # Select card based on exploration strategy
        if self.config.exploration_strategy == ExplorationStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy_action(card_logits, placement, state)
        elif self.config.exploration_strategy == ExplorationStrategy.BOLTZMANN:
            return self._boltzmann_action(card_logits, placement, state, temperature)
        elif self.config.exploration_strategy == ExplorationStrategy.ENTROPY_REGULARIZED:
            return self._entropy_action(card_logits, placement, state)
        else:
            return self._epsilon_greedy_action(card_logits, placement, state)

    def _apply_action_mask(self, logits: np.ndarray,
                           valid_cards: np.ndarray) -> np.ndarray:
        """Apply action mask to logits, setting invalid card logits to -inf."""
        masked = logits.copy()
        # Set invalid cards to very negative value
        for i in range(len(valid_cards)):
            if not valid_cards[i]:
                masked[i] = -1e9
        # PASS is always valid
        masked[4] = max(masked[4], 0.0)  # Ensure pass has at least neutral logit
        return masked

    def _epsilon_greedy_action(self, card_logits: np.ndarray,
                                placement: np.ndarray,
                                state: np.ndarray) -> dict:
        """Select action using epsilon-greedy exploration."""
        is_exploration = False

        if self.rng.random() < self.epsilon:
            is_exploration = True
            self.exploration_actions += 1
            # Random valid card
            card_idx = self.rng.randint(0, 5)  # 0-3 cards + pass
            # Random placement
            target_col = float(self.rng.randint(0, self.config.grid_cols))
            target_row = float(self.rng.randint(0, self.config.grid_rows))
        else:
            # Greedy: argmax of logits
            card_idx = int(np.argmax(card_logits))
            # Use network placement
            target_col = (placement[0] + 1) / 2 * self.config.grid_cols
            target_row = (placement[1] + 1) / 2 * self.config.grid_rows
            target_col = max(0, min(self.config.grid_cols - 1, target_col))
            target_row = max(0, min(self.config.grid_rows - 1, target_row))

        # Compute softmax probabilities for logging
        probs = self._softmax(card_logits)

        return {
            "card_idx": card_idx,
            "target_col": float(target_col),
            "target_row": float(target_row),
            "is_exploration": is_exploration,
            "card_logits": card_logits.tolist(),
            "card_probs": probs.tolist(),
        }

    def _boltzmann_action(self, card_logits: np.ndarray,
                          placement: np.ndarray,
                          state: np.ndarray,
                          temperature: Optional[float] = None) -> dict:
        """Select action using Boltzmann (softmax) exploration."""
        temp = temperature or self.config.temperature
        probs = self._softmax(card_logits / temp)

        # Sample card from distribution
        card_idx = self.rng.choice(5, p=probs)
        is_exploration = card_idx != np.argmax(card_logits)

        if is_exploration:
            self.exploration_actions += 1

        # Use network placement
        target_col = (placement[0] + 1) / 2 * self.config.grid_cols
        target_row = (placement[1] + 1) / 2 * self.config.grid_rows
        target_col = max(0, min(self.config.grid_cols - 1, target_col))
        target_row = max(0, min(self.config.grid_rows - 1, target_row))

        return {
            "card_idx": card_idx,
            "target_col": float(target_col),
            "target_row": float(target_row),
            "is_exploration": is_exploration,
            "card_logits": card_logits.tolist(),
            "card_probs": probs.tolist(),
        }

    def _entropy_action(self, card_logits: np.ndarray,
                        placement: np.ndarray,
                        state: np.ndarray) -> dict:
        """Select action with entropy regularization bonus."""
        probs = self._softmax(card_logits)
        
        # Compute entropy of the distribution
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Add entropy bonus to exploration
        if self.rng.random() < self.epsilon * (1 + self.config.entropy_coeff * entropy):
            # Explore: sample from distribution
            card_idx = self.rng.choice(5, p=probs)
            is_exploration = True
            self.exploration_actions += 1
        else:
            # Exploit: argmax
            card_idx = int(np.argmax(card_logits))
            is_exploration = False

        target_col = (placement[0] + 1) / 2 * self.config.grid_cols
        target_row = (placement[1] + 1) / 2 * self.config.grid_rows
        target_col = max(0, min(self.config.grid_cols - 1, target_col))
        target_row = max(0, min(self.config.grid_rows - 1, target_row))

        return {
            "card_idx": card_idx,
            "target_col": float(target_col),
            "target_row": float(target_row),
            "is_exploration": is_exploration,
            "card_logits": card_logits.tolist(),
            "card_probs": probs.tolist(),
        }

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        # Subtract max for numerical stability
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        return exp_shifted / (np.sum(exp_shifted) + 1e-10)

    def get_weights(self) -> np.ndarray:
        """Get the current network weights."""
        return self.network.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the network weights."""
        self.network.set_weights(weights)

    def decay_epsilon(self, factor: Optional[float] = None) -> float:
        """Decay exploration rate.

        Args:
            factor: Custom decay factor. Uses config default if None.

        Returns:
            Current epsilon value.
        """
        f = factor if factor is not None else self.config.epsilon_decay
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * f,
        )
        return self.epsilon

    def reset_epsilon(self) -> None:
        """Reset epsilon to initial value."""
        self.epsilon = self.config.epsilon

    def save_checkpoint(self, path: str) -> None:
        """Save the agent to a file.

        The evolved policy genome is what decides how the agent plays, so it
        is persisted whenever the agent carries one. Torch parameters are only
        written if the network was actually built -- otherwise this would save
        a freshly-initialised (untrained) network and, worse, restore one on
        load, silently discarding the result of a training run.

        Args:
            path: File path to save to.
        """
        genome = None if self.weights is None else np.asarray(self.weights)
        payload = {
            "genome": genome,
            # "weights" is retained for readers that predate the genome split.
            "weights": genome if genome is not None else self.get_weights(),
            "param_kind": "genome" if genome is not None else "network",
            "network_weights": self.get_weights() if self.has_network else None,
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "exploration_actions": self.exploration_actions,
            "best_fitness": self.best_fitness,
            "best_weights": self.best_weights,
            "config": {
                "architecture": self.config.architecture,
                "input_channels": self.config.input_channels,
                "input_height": self.config.input_height,
                "input_width": self.config.input_width,
                "hidden_size": self.config.hidden_size,
                "lstm_layers": self.config.lstm_layers,
                "temperature": self.config.temperature,
                "exploration_strategy": self.config.exploration_strategy,
            },
        }
        torch.save(payload, path)
        logger.info(f"Saved agent checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load an agent from a file.

        Restores the evolved genome when the checkpoint holds one, and only
        touches the Torch network for checkpoints that actually carry network
        parameters.

        Args:
            path: File path to load from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        stored = checkpoint.get("weights")
        genome = checkpoint.get("genome")
        kind = checkpoint.get("param_kind")
        if kind is None:
            # Written before the genome/network split: infer from length.
            kind = "genome" if _looks_like_genome(stored) else "network"
        if genome is None and kind == "genome":
            genome = stored

        if genome is not None:
            self.weights = np.asarray(genome)
        network_weights = checkpoint.get("network_weights")
        if network_weights is None and kind == "network":
            network_weights = stored
        if network_weights is not None:
            self.set_weights(network_weights)

        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.total_actions = checkpoint.get("total_actions", 0)
        self.exploration_actions = checkpoint.get("exploration_actions", 0)
        self.best_fitness = checkpoint.get("best_fitness", -float('inf'))
        self.best_weights = checkpoint.get("best_weights")
        config_data = checkpoint.get("config", {})
        if config_data:
            self.config.architecture = config_data.get("architecture", self.config.architecture)
            self.config.temperature = config_data.get("temperature", self.config.temperature)
            self.config.exploration_strategy = config_data.get("exploration_strategy", self.config.exploration_strategy)
        logger.info(f"Loaded agent checkpoint from {path}")

    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.epsilon

    def reset(self) -> None:
        """Reset LSTM hidden state (for CNN+LSTM agents)."""
        if hasattr(self.network, "reset"):
            self.network.reset()

    def update_best(self, fitness: float) -> None:
        """Update best fitness tracking.

        Args:
            fitness: Current fitness score.
        """
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_weights = self.get_weights().copy()

    def __repr__(self) -> str:
        return (f"EvolutionaryAgent(arch={self.config.architecture}, "
                f"epsilon={self.epsilon:.4f}, "
                f"weights_shape={self.get_weights().shape}, "
                f"strategy={self.config.exploration_strategy})")
