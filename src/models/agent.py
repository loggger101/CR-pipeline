"""Evolutionary agent wrapper.

Encapsulates a neural network with action selection, exploration,
and weight management capabilities. Supports both greedy inference
and epsilon-greedy exploration.
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


@dataclass
class AgentConfig:
    """Configuration for an evolutionary agent.

    Attributes:
        architecture: Type of neural network architecture.
        hidden_size: Size of hidden layers.
        lstm_layers: Number of LSTM layers (for CNN+LSTM).
        epsilon: Exploration rate (epsilon-greedy).
        epsilon_decay: Rate at which epsilon decays.
        epsilon_min: Minimum epsilon value.
        grid_cols: Grid columns for placement.
        grid_rows: Grid rows for placement.
        device: Device for inference.
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
    - Action selection with exploration
    - Weight management (get/set)
    - Checkpoint save/load
    - Epsilon decay for exploration

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

        # Create network
        if self.config.architecture == "cnn_lstm":
            self.network = create_cnn_lstm_agent(
                input_channels=self.config.input_channels,
                input_height=self.config.input_height,
                input_width=self.config.input_width,
                hidden_size=self.config.hidden_size,
                lstm_layers=self.config.lstm_layers,
                device=self.config.device,
            )
        else:
            self.network = create_cnn_mlp_agent(
                input_channels=self.config.input_channels,
                input_height=self.config.input_height,
                input_width=self.config.input_width,
                hidden_size=self.config.hidden_size,
                device=self.config.device,
            )

        # Exploration
        self.epsilon = self.config.epsilon
        self.weights = weights

        # Statistics
        self.total_actions = 0
        self.exploration_actions = 0

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

        return card_logits.cpu().numpy(), placement.cpu().numpy()

    def select_action(self, state: np.ndarray,
                      valid_cards: Optional[np.ndarray] = None,
                      valid_positions: Optional[np.ndarray] = None) -> dict:
        """Select an action with exploration.

        Args:
            state: Input state tensor.
            valid_cards: Boolean mask of valid card indices (4,).
            valid_positions: Boolean mask of valid positions.

        Returns:
            Action dict with:
            - card_idx: Selected card index (-1 for pass)
            - target_col: Target column
            - target_row: Target row
            - is_exploration: Whether this was an exploratory action
        """
        self.total_actions += 1

        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            self.exploration_actions += 1
            return self._exploration_action(state)
        else:
            return self._greedy_action(state, valid_cards)

    def _greedy_action(self, state: np.ndarray,
                       valid_cards: Optional[np.ndarray] = None) -> dict:
        """Select the greedy (best) action."""
        card_logits, placement = self.forward(state)

        # Select card with highest logit
        card_idx = int(np.argmax(card_logits))

        # Clip placement to grid
        target_col = float(placement[0])
        target_row = float(placement[1])

        # Normalize from [-1, 1] to [0, grid_size]
        target_col = (target_col + 1) / 2 * self.config.grid_cols
        target_row = (target_row + 1) / 2 * self.config.grid_rows

        target_col = max(0, min(self.config.grid_cols - 1, target_col))
        target_row = max(0, min(self.config.grid_rows - 1, target_row))

        return {
            "card_idx": card_idx,
            "target_col": float(target_col),
            "target_row": float(target_row),
            "is_exploration": False,
            "card_logits": card_logits.tolist(),
        }

    def _exploration_action(self, state: np.ndarray) -> dict:
        """Select a random exploratory action."""
        # Random card index (0-3) or pass (-1)
        card_idx = self.rng.randint(-1, 4)

        # Random placement
        target_col = float(self.rng.randint(0, self.config.grid_cols))
        target_row = float(self.rng.randint(0, self.config.grid_rows))

        return {
            "card_idx": card_idx,
            "target_col": target_col,
            "target_row": target_row,
            "is_exploration": True,
            "card_logits": None,
        }

    def get_weights(self) -> np.ndarray:
        """Get the current network weights."""
        return self.network.get_weights()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the network weights."""
        self.network.set_weights(weights)

    def decay_epsilon(self) -> float:
        """Decay exploration rate.

        Returns:
            Current epsilon value.
        """
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )
        return self.epsilon

    def save_checkpoint(self, path: str) -> None:
        """Save agent weights to a file.

        Args:
            path: File path to save to.
        """
        torch.save({
            "weights": self.get_weights(),
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "exploration_actions": self.exploration_actions,
        }, path)
        logger.info(f"Saved agent checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent weights from a file.

        Args:
            path: File path to load from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.set_weights(checkpoint["weights"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.total_actions = checkpoint.get("total_actions", 0)
        self.exploration_actions = checkpoint.get("exploration_actions", 0)
        logger.info(f"Loaded agent checkpoint from {path}")

    def get_exploration_rate(self) -> float:
        """Get current exploration rate."""
        return self.epsilon

    def reset(self) -> None:
        """Reset LSTM hidden state (for CNN+LSTM agents)."""
        if hasattr(self.network, "reset"):
            self.network.reset()

    def __repr__(self) -> str:
        return (f"EvolutionaryAgent(arch={self.config.architecture}, "
                f"epsilon={self.epsilon:.4f}, "
                f"weights_shape={self.get_weights().shape})")
