"""Neural network architectures for Clash Royale agents.

Provides two main architectures:
1. CNN+LSTM: Frame-based input processed by convolutions, then temporal
   modeling via LSTM. Suitable for capturing game dynamics.
2. CNN+MLP: Two-stream network processing visual state and game state
   features separately, then concatenating for action prediction.

All networks output:
- Card selection logits: 5 values (4 cards + pass)
- Placement coordinates: 2 continuous values (col, row)
Total output dim: 7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AgentArchitecture:
    """Configuration for an agent network architecture.

    Attributes:
        name: Architecture identifier.
        input_channels: Number of input channels (state tensor channels).
        input_height: Height of input tensor.
        input_width: Width of input tensor.
        hidden_size: Size of hidden layers.
        lstm_layers: Number of LSTM layers.
        num_cards: Number of cards in hand.
        grid_cols: Grid columns for placement.
        grid_rows: Grid rows for placement.
        action_dim: Total action dimension (cards + 2 placement).
    """
    name: str
    input_channels: int
    input_height: int
    input_width: int
    hidden_size: int = 256
    lstm_layers: int = 2
    num_cards: int = 4
    grid_cols: int = 8
    grid_rows: int = 6
    action_dim: int = 7  # 4 cards + pass + 2 placement coords

    @property
    def card_logits_dim(self) -> int:
        """Number of card selection logits."""
        return self.num_cards + 1  # cards + pass


class CNNLSTMAgent(nn.Module):
    """CNN + LSTM architecture for frame-based input.

    Processes the state tensor through convolutions to extract spatial
    features, then uses an LSTM to model temporal dependencies across
    game ticks.

    Input:  (batch, channels, H, W)
    Output: (batch, card_logits_dim + 2)
            card_logits_dim: logits for card selection
            2: continuous placement coordinates
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.hidden_size = arch.hidden_size

        # CNN feature extractor
        self.conv1 = nn.Conv2d(arch.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling
        pool_size = arch.input_height // 8  # After 3 maxpools
        cnn_output = 128 * pool_size * arch.input_width // 8

        # LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output,
            hidden_size=self.hidden_size,
            num_layers=arch.lstm_layers,
            batch_first=True,
            dropout=0.1 if arch.lstm_layers > 1 else 0,
        )

        # Action heads
        self.card_head = nn.Linear(self.hidden_size, arch.card_logits_dim)
        self.pos_head = nn.Linear(self.hidden_size, 2)

        # LSTM state
        self.hidden_state = None
        self.cell_state = None

    def forward(self, x: torch.Tensor,
                reset: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, H, W).
            reset: Whether to reset LSTM hidden state.

        Returns:
            card_logits: (batch, card_logits_dim)
            placement: (batch, 2) continuous [col, row]
        """
        batch_size = x.shape[0]

        # CNN
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, 2)

        # Flatten
        h = h.view(batch_size, -1)

        # LSTM (process as single timestep)
        h = h.unsqueeze(1)  # (batch, 1, features)

        if self.hidden_state is None or reset:
            self.hidden_state = torch.zeros(
                self.arch.lstm_layers, batch_size, self.hidden_size,
                device=x.device,
            )
            self.cell_state = torch.zeros(
                self.arch.lstm_layers, batch_size, self.hidden_size,
                device=x.device,
            )

        lstm_out, (hn, _) = self.lstm(h, (self.hidden_state, self.cell_state))
        lstm_out = lstm_out.squeeze(1)  # (batch, hidden)

        # Action heads
        card_logits = self.card_head(lstm_out)
        placement = torch.tanh(self.pos_head(lstm_out))  # [-1, 1]

        return card_logits, placement

    def reset(self) -> None:
        """Reset LSTM hidden state."""
        self.hidden_state = None
        self.cell_state = None

    def get_weights(self) -> np.ndarray:
        """Extract network weights as numpy array."""
        return torch.cat([p.detach().flatten() for p in self.parameters()]).cpu().numpy()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set network weights from numpy array."""
        flat = torch.tensor(weights, dtype=torch.float32)
        idx = 0
        for param in self.parameters():
            size = param.numel()
            if idx + size > len(flat):
                raise ValueError(f"Weight size mismatch: need {size}, have {len(flat) - idx}")
            param.data = flat[idx:idx + size].view(param.shape).clone()
            idx += size


class CNNMLPAgent(nn.Module):
    """CNN + MLP two-stream architecture for state-augmented input.

    Processes visual features and game state features separately,
    then concatenates for action prediction. Suitable for when the
    state representation includes explicit game features.

    Input:  (batch, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch

        # Vision branch
        self.vision_conv1 = nn.Conv2d(arch.input_channels, 32, kernel_size=3, padding=1)
        self.vision_bn1 = nn.BatchNorm2d(32)
        self.vision_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.vision_bn2 = nn.BatchNorm2d(64)
        self.vision_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.vision_bn3 = nn.BatchNorm2d(128)

        pool_size = arch.input_height // 8
        vision_dim = 128 * pool_size * (arch.input_width // 8)

        # State branch (for explicit game features like elixir, time)
        self.state_fc1 = nn.Linear(arch.input_channels, 64)
        self.state_fc2 = nn.Linear(64, 64)

        # Combined head
        combined_dim = vision_dim + 64
        self.combined_fc1 = nn.Linear(combined_dim, arch.hidden_size)
        self.combined_fc2 = nn.Linear(arch.hidden_size, arch.hidden_size // 2)

        # Action heads
        self.card_head = nn.Linear(arch.hidden_size // 2, arch.card_logits_dim)
        self.pos_head = nn.Linear(arch.hidden_size // 2, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, H, W).

        Returns:
            card_logits: (batch, card_logits_dim)
            placement: (batch, 2) continuous [col, row]
        """
        batch_size = x.shape[0]

        # Vision branch
        v = F.relu(self.vision_bn1(self.vision_conv1(x)))
        v = F.max_pool2d(v, 2)
        v = F.relu(self.vision_bn2(self.vision_conv2(v)))
        v = F.max_pool2d(v, 2)
        v = F.relu(self.vision_bn3(self.vision_conv3(v)))
        v = F.max_pool2d(v, 2)
        v = v.view(batch_size, -1)

        # State branch (use mean of channels as features)
        s = x.mean(dim=[2, 3])  # (batch, channels)
        s = F.relu(self.state_fc1(s))
        s = F.relu(self.state_fc2(s))

        # Combine
        combined = torch.cat([v, s], dim=1)
        h = F.relu(self.combined_fc1(combined))
        h = F.relu(self.combined_fc2(h))

        # Action heads
        card_logits = self.card_head(h)
        placement = torch.tanh(self.pos_head(h))

        return card_logits, placement

    def get_weights(self) -> np.ndarray:
        """Extract network weights as numpy array."""
        return torch.cat([p.detach().flatten() for p in self.parameters()]).cpu().numpy()

    def set_weights(self, weights: np.ndarray) -> None:
        """Set network weights from numpy array."""
        flat = torch.tensor(weights, dtype=torch.float32)
        idx = 0
        for param in self.parameters():
            size = param.numel()
            if idx + size > len(flat):
                raise ValueError(f"Weight size mismatch: need {size}, have {len(flat) - idx}")
            param.data = flat[idx:idx + size].view(param.shape).clone()
            idx += size


def create_cnn_lstm_agent(input_channels: int = 12,
                          input_height: int = 64,
                          input_width: int = 64,
                          hidden_size: int = 256,
                          lstm_layers: int = 2,
                          device: str = "cpu") -> CNNLSTMAgent:
    """Create a CNN+LSTM agent.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        device: Device to place the network on.

    Returns:
        CNNLSTMAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_lstm",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )
    agent = CNNLSTMAgent(arch).to(device)
    return agent


def create_cnn_mlp_agent(input_channels: int = 12,
                         input_height: int = 64,
                         input_width: int = 64,
                         hidden_size: int = 256,
                         device: str = "cpu") -> CNNMLPAgent:
    """Create a CNN+MLP agent.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        device: Device to place the network on.

    Returns:
        CNNMLPAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_mlp",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
    )
    agent = CNNMLPAgent(arch).to(device)
    return agent
