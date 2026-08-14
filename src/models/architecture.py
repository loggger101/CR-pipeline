"""Neural network architectures for Clash Royale agents.

Provides multiple architectures:
1. CNN+LSTM: Frame-based input processed by convolutions, then temporal
   modeling via LSTM. Suitable for capturing game dynamics.
2. CNN+MLP: Two-stream network processing visual state and game state
   features separately, then concatenating for action prediction.
3. CNN+ResNet: Residual network for deeper feature extraction.
4. CNN+Transformer: Vision transformer for global attention over the state.

All networks output:
- Card selection logits: 5 values (4 cards + pass)
- Placement coordinates: 2 continuous values (col, row)
Total output dim: 7
"""

from __future__ import annotations

import math
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
        nheads: Number of attention heads (for Transformer).
        transformer_layers: Number of transformer encoder layers.
        dropout: Dropout rate.
        residual: Whether to use residual connections.
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
    nheads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1
    residual: bool = True

    @property
    def card_logits_dim(self) -> int:
        """Number of card selection logits."""
        return self.num_cards + 1  # cards + pass


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and batch norm."""

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.training:
            out = F.dropout(out, p=0.1, inplace=True)
        return F.relu(out + identity)


class CNNResNetAgent(nn.Module):
    """CNN with residual blocks for deeper feature extraction.

    Uses residual blocks to enable deeper networks without vanishing gradients.
    Suitable for when more complex visual patterns need to be learned.

    Input:  (batch, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.hidden_size = arch.hidden_size

        # Initial conv
        self.conv_init = nn.Conv2d(arch.input_channels, 32, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(32)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(32),
            ResidualBlock(64),
            ResidualBlock(128),
        ])

        # Pooling
        pool_size = arch.input_height // 8
        cnn_output = 128 * pool_size * (arch.input_width // 8)

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

        # Initial conv
        h = F.relu(self.bn_init(self.conv_init(x)))
        h = F.max_pool2d(h, 2)

        # Residual blocks with pooling between
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h)
            if i < len(self.res_blocks) - 1:
                h = F.max_pool2d(h, 2)

        # Flatten
        h = h.view(batch_size, -1)

        # LSTM
        h = h.unsqueeze(1)

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
        lstm_out = lstm_out.squeeze(1)

        # Action heads
        card_logits = self.card_head(lstm_out)
        placement = torch.tanh(self.pos_head(lstm_out))

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


class CNNTransformerAgent(nn.Module):
    """CNN feature extractor + Transformer encoder.

    Uses a Vision Transformer approach where the CNN extracts local features
    and the Transformer provides global attention over the spatial features.
    Better at capturing long-range dependencies in the game state.

    Input:  (batch, channels, H, W)
    Output: (batch, card_logits_dim + 2)
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

        # Pooling to get spatial features
        pool_size = arch.input_height // 8
        self.spatial_h = pool_size
        self.spatial_w = arch.input_width // 8
        self.feature_dim = 128

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.spatial_h * self.spatial_w, self.feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=arch.nheads,
            dim_feedforward=arch.hidden_size,
            dropout=arch.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=arch.transformer_layers)

        # Action heads
        self.card_head = nn.Linear(self.feature_dim, arch.card_logits_dim)
        self.pos_head = nn.Linear(self.feature_dim, 2)

        # LSTM for temporal modeling over transformer output
        self.temporal_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
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

        # Reshape to sequence: (batch, seq_len, feature_dim)
        h = h.permute(0, 2, 3, 1).reshape(batch_size, self.spatial_h * self.spatial_w, self.feature_dim)

        # Add positional encoding
        h = h + self.pos_encoding

        # Transformer
        h = self.transformer(h)

        # Average pooling over spatial dimensions
        h = h.mean(dim=1)  # (batch, feature_dim)

        # Temporal LSTM
        h = h.unsqueeze(1)

        if self.hidden_state is None or reset:
            self.hidden_state = torch.zeros(
                1, batch_size, self.hidden_size,
                device=x.device,
            )
            self.cell_state = torch.zeros(
                1, batch_size, self.hidden_size,
                device=x.device,
            )

        lstm_out, (hn, _) = self.temporal_lstm(h, (self.hidden_state, self.cell_state))
        lstm_out = lstm_out.squeeze(1)

        # Action heads
        card_logits = self.card_head(lstm_out)
        placement = torch.tanh(self.pos_head(lstm_out))

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




class CNNCNNMLPAgent(nn.Module):
    """Dual-stream CNN processing visual and state features separately.

    Uses two parallel CNN branches for visual state and game state,
    then concatenates for action prediction.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(arch.input_channels // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.state_cnn = nn.Sequential(
            nn.Conv2d(arch.input_channels // 2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16 * 2, 512),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(256, arch.card_logits_dim + 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        visual = x[:, :, :channels // 2, :, :]
        state = x[:, :, channels // 2:, :, :]
        visual_feat = self.visual_cnn(visual.view(batch * seq_len, channels // 2, h, w))
        state_feat = self.state_cnn(state.view(batch * seq_len, channels // 2, h, w))
        visual_feat = visual_feat.view(batch, seq_len, -1)
        state_feat = state_feat.view(batch, seq_len, -1)
        combined = torch.cat([visual_feat, state_feat], dim=-1)
        out = self.fc(combined)
        return out[:, -1, :]


class CNNGRUAgent(nn.Module):
    """CNN with GRU for lighter temporal modeling.

    GRU is faster than LSTM with comparable performance on many tasks.
    Suitable when training speed is a concern.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.cnn = nn.Sequential(
            nn.Conv2d(arch.input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spatial_dim = 128 * 4 * 4
        self.gru = nn.GRU(self.spatial_dim, arch.hidden_size, arch.lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(arch.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(128, arch.card_logits_dim + 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        x = x.view(batch * seq_len, channels, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq_len, self.spatial_dim)
        out, _ = self.gru(features)
        return self.fc(out[:, -1, :])


class CNNLSTMAttentionAgent(nn.Module):
    """CNN+LSTM with attention over time steps.

    Uses attention mechanism to weight the importance of different time steps.
    Useful when some game states are more informative than others.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.cnn = nn.Sequential(
            nn.Conv2d(arch.input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spatial_dim = 128 * 4 * 4
        self.lstm = nn.LSTM(self.spatial_dim, arch.hidden_size, arch.lstm_layers, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(arch.hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(arch.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(128, arch.card_logits_dim + 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        x = x.view(batch * seq_len, channels, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq_len, self.spatial_dim)
        lstm_out, _ = self.lstm(features)
        # Attention weights
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        # Weighted sum
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        return self.fc(context)


class CNNResNetLSTMAgent(nn.Module):
    """CNN+ResNet+LSTM for deep feature extraction with temporal modeling.

    Combines ResNet residual blocks with LSTM for both deep feature extraction
    and temporal modeling.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.cnn = nn.Sequential(
            nn.Conv2d(arch.input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.resnet = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            nn.MaxPool2d(2),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.spatial_dim = 128 * 2 * 2
        self.lstm = nn.LSTM(self.spatial_dim, arch.hidden_size, arch.lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(arch.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(128, arch.card_logits_dim + 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        x = x.view(batch * seq_len, channels, h, w)
        features = self.cnn(x)
        features = self.resnet(features)
        features = features.view(batch, seq_len, self.spatial_dim)
        out, _ = self.lstm(features)
        return self.fc(out[:, -1, :])


class CNNTransformerLSTMAgent(nn.Module):
    """CNN+Transformer+LSTM for global attention with temporal modeling.

    Uses Transformer for spatial attention and LSTM for temporal modeling.
    Best for complex games requiring both spatial and temporal reasoning.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.cnn = nn.Sequential(
            nn.Conv2d(arch.input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.spatial_dim = 256 * 4 * 4
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=arch.nheads, dim_feedforward=512, dropout=arch.dropout),
            num_layers=arch.transformer_layers,
        )
        self.temporal = nn.LSTM(256, arch.hidden_size, arch.lstm_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(arch.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(arch.dropout),
            nn.Linear(128, arch.card_logits_dim + 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        x = x.view(batch * seq_len, channels, h, w)
        features = self.cnn(x)
        features = features.view(batch, seq_len, self.spatial_dim)
        features = self.transformer(features)
        features = features[:, -1, :]
        out, _ = self.temporal(features.unsqueeze(1))
        out = out[:, -1, :]
        return self.fc(out)


class CNNConvLSTMAgent(nn.Module):
    """CNN+ConvLSTM for spatiotemporal feature extraction.

    ConvLSTM captures both spatial and temporal patterns in game state frames,
    making it ideal for understanding dynamic battle situations where the
    spatial arrangement of units evolves over time.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
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
        self.pool = nn.MaxPool2d(2)

        # ConvLSTM layers (simulated with LSTM on flattened spatial features)
        pool_size = arch.input_height // 4
        cnn_output = 64 * pool_size * (arch.input_width // 4)
        self.spatial_proj = nn.Linear(cnn_output, arch.hidden_size)

        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=arch.hidden_size,
            hidden_size=arch.hidden_size,
            num_layers=arch.lstm_layers,
            batch_first=True,
            dropout=0.1 if arch.lstm_layers > 1 else 0,
        )

        # Action heads
        self.card_head = nn.Linear(arch.hidden_size, arch.card_logits_dim)
        self.pos_head = nn.Linear(arch.hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        x = x.view(batch * seq_len, channels, h, w)
        
        # CNN features
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool(out)
        
        # Flatten spatial features
        out = out.view(batch, seq_len, -1)
        out = self.spatial_proj(out)
        
        # Temporal modeling
        out, _ = self.lstm(out)
        return torch.cat([self.card_head(out[:, -1, :]), self.pos_head(out[:, -1, :])], dim=-1)


class CNNCRNNAgent(nn.Module):
    """CNN+CRNN (Convolutional RNN) for sequence modeling.

    Applies convolutions along the sequence dimension to capture local temporal
    patterns efficiently. Useful for identifying short-term tactical patterns
    in unit movements and engagements.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.hidden_size = arch.hidden_size

        # CNN feature extractor per timestep
        self.conv1 = nn.Conv2d(arch.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)

        pool_size = arch.input_height // 2
        cnn_output = 32 * pool_size * (arch.input_width // 2)

        # CRNN: apply convolutions across sequence dimension
        self.seq_conv = nn.Conv1d(cnn_output, arch.hidden_size, kernel_size=3, padding=1)
        self.seq_bn = nn.BatchNorm1d(arch.hidden_size)

        # Final LSTM for temporal context
        self.lstm = nn.LSTM(
            input_size=arch.hidden_size,
            hidden_size=arch.hidden_size,
            num_layers=arch.lstm_layers,
            batch_first=True,
            dropout=0.1 if arch.lstm_layers > 1 else 0,
        )

        # Action heads
        self.card_head = nn.Linear(arch.hidden_size, arch.card_logits_dim)
        self.pos_head = nn.Linear(arch.hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        
        # Apply CNN to each timestep independently
        features = []
        for t in range(seq_len):
            frame = x[:, t]  # (batch, channels, H, W)
            out = F.relu(self.bn1(self.conv1(frame)))
            out = self.pool(out)
            out = out.view(batch, -1)  # Flatten spatial
            features.append(out)
        
        features = torch.stack(features, dim=1)  # (batch, seq_len, cnn_output)
        
        # Conv1D across sequence dimension
        features = features.permute(0, 2, 1)  # (batch, cnn_output, seq_len)
        out = F.relu(self.seq_bn(self.seq_conv(features)))
        out = out.permute(0, 2, 1)  # (batch, seq_len, hidden_size)
        
        # LSTM
        out, _ = self.lstm(out)
        return torch.cat([self.card_head(out[:, -1, :]), self.pos_head(out[:, -1, :])], dim=-1)


class DuelingAgent(nn.Module):
    """Dueling DQN-style agent for action-value estimation.

    Separates state value and advantage streams to better estimate which
    actions are valuable regardless of the current state value.

    Input:  (batch, seq_len, channels, H, W)
    Output: (batch, card_logits_dim + 2)
    """

    def __init__(self, arch: AgentArchitecture):
        super().__init__()
        self.arch = arch
        self.hidden_size = arch.hidden_size

        # Shared CNN backbone
        self.conv1 = nn.Conv2d(arch.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        pool_size = arch.input_height // 4
        cnn_output = 64 * pool_size * (arch.input_width // 4)

        # Shared layers
        self.shared_fc1 = nn.Linear(cnn_output, arch.hidden_size)
        self.shared_bn = nn.BatchNorm1d(arch.hidden_size)

        # Value stream
        self.value_fc = nn.Linear(arch.hidden_size, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(arch.hidden_size, arch.card_logits_dim + 2)

        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=arch.hidden_size,
            hidden_size=arch.hidden_size,
            num_layers=arch.lstm_layers,
            batch_first=True,
            dropout=0.1 if arch.lstm_layers > 1 else 0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels, h, w = x.shape
        
        # CNN features per timestep
        frame_features = []
        for t in range(seq_len):
            frame = x[:, t]
            out = F.relu(self.bn1(self.conv1(frame)))
            out = self.pool(out)
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.pool(out)
            out = out.view(batch, -1)
            frame_features.append(out)
        
        # Stack and process through shared layers
        features = torch.stack(frame_features, dim=1)  # (batch, seq_len, cnn_output)
        features = F.relu(self.shared_bn(self.shared_fc1(features)))
        
        # LSTM for temporal context
        features, _ = self.lstm(features)
        last_feat = features[:, -1, :]  # (batch, hidden_size)
        
        # Dueling: separate value and advantage
        value = self.value_fc(last_feat).unsqueeze(-1)  # (batch, 1, 1)
        advantages = self.advantage_fc(last_feat)  # (batch, card_logits_dim + 2)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        mean_adv = advantages.mean(dim=-1, keepdim=True)
        q_values = value + (advantages - mean_adv)
        
        return q_values.squeeze(-1)


class EnsembleAgent(nn.Module):
    """Ensemble of agents for improved prediction stability.

    Combines predictions from multiple agent models using weighted averaging,
    producing more stable and generalizable predictions than any single agent.
    """

    def __init__(self, agent_list: list, weights: Optional[list] = None):
        super().__init__()
        self.agents = nn.ModuleList(agent_list)
        n = len(agent_list)
        if weights is None:
            weights = [1.0 / n] * n
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect predictions from all agents
        predictions = []
        for agent in self.agents:
            pred = agent(x)
            predictions.append(pred)
        
        # Weighted average
        predictions = torch.stack(predictions, dim=-1)  # (batch, output_dim, n_agents)
        weights = self.weights.view(1, 1, -1)
        result = (predictions * weights).sum(dim=-1)
        return result


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


def create_cnn_resnet_agent(input_channels: int = 12,
                            input_height: int = 64,
                            input_width: int = 64,
                            hidden_size: int = 256,
                            lstm_layers: int = 2,
                            device: str = "cpu") -> CNNResNetAgent:
    """Create a CNN+ResNet agent with residual blocks.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        device: Device to place the network on.

    Returns:
        CNNResNetAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_resnet",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )
    agent = CNNResNetAgent(arch).to(device)
    return agent


def create_cnn_transformer_agent(input_channels: int = 12,
                                 input_height: int = 64,
                                 input_width: int = 64,
                                 hidden_size: int = 256,
                                 nheads: int = 4,
                                 transformer_layers: int = 2,
                                 device: str = "cpu") -> CNNTransformerAgent:
    """Create a CNN+Transformer agent with attention over spatial features.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        nheads: Number of attention heads.
        transformer_layers: Number of transformer encoder layers.
        device: Device to place the network on.

    Returns:
        CNNTransformerAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_transformer",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        nheads=nheads,
        transformer_layers=transformer_layers,
    )
    agent = CNNTransformerAgent(arch).to(device)
    return agent


def create_cnn_cnn_mlp_agent(input_channels: int = 12,
                             input_height: int = 64,
                             input_width: int = 64,
                             hidden_size: int = 256,
                             device: str = "cpu") -> CNNCNNMLPAgent:
    """Create a dual-stream CNN+MLP agent.

    Uses two parallel CNN branches for visual and state features.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        device: Device to place the network on.

    Returns:
        CNNCNNMLPAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_cnn_mlp",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
    )
    agent = CNNCNNMLPAgent(arch).to(device)
    return agent


def create_cnn_gru_agent(input_channels: int = 12,
                         input_height: int = 64,
                         input_width: int = 64,
                         hidden_size: int = 256,
                         gru_layers: int = 2,
                         device: str = "cpu") -> CNNGRUAgent:
    """Create a CNN+GRU agent for faster temporal modeling.

    GRU is faster than LSTM with comparable performance.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: GRU hidden size.
        gru_layers: Number of GRU layers.
        device: Device to place the network on.

    Returns:
        CNNGRUAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_gru",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=gru_layers,
    )
    agent = CNNGRUAgent(arch).to(device)
    return agent


def create_cnn_lstm_attention_agent(input_channels: int = 12,
                                    input_height: int = 64,
                                    input_width: int = 64,
                                    hidden_size: int = 256,
                                    lstm_layers: int = 2,
                                    device: str = "cpu") -> CNNLSTMAttentionAgent:
    """Create a CNN+LSTM with attention agent.

    Uses attention to weight the importance of different time steps.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        device: Device to place the network on.

    Returns:
        CNNLSTMAttentionAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_lstm_attention",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )
    agent = CNNLSTMAttentionAgent(arch).to(device)
    return agent


def create_cnn_resnet_lstm_agent(input_channels: int = 12,
                                 input_height: int = 64,
                                 input_width: int = 64,
                                 hidden_size: int = 256,
                                 lstm_layers: int = 2,
                                 device: str = "cpu") -> CNNResNetLSTMAgent:
    """Create a CNN+ResNet+LSTM agent.

    Combines ResNet residual blocks with LSTM.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: LSTM hidden size.
        lstm_layers: Number of LSTM layers.
        device: Device to place the network on.

    Returns:
        CNNResNetLSTMAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_resnet_lstm",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )
    agent = CNNResNetLSTMAgent(arch).to(device)
    return agent


def create_cnn_transformer_lstm_agent(input_channels: int = 12,
                                      input_height: int = 64,
                                      input_width: int = 64,
                                      hidden_size: int = 256,
                                      nheads: int = 4,
                                      transformer_layers: int = 2,
                                      device: str = "cpu") -> CNNTransformerLSTMAgent:
    """Create a CNN+Transformer+LSTM agent.

    Transformer for spatial attention, LSTM for temporal modeling.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: LSTM hidden size.
        nheads: Number of attention heads.
        transformer_layers: Number of transformer encoder layers.
        device: Device to place the network on.

    Returns:
        CNNTransformerLSTMAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_transformer_lstm",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        nheads=nheads,
        transformer_layers=transformer_layers,
    )
    agent = CNNTransformerLSTMAgent(arch).to(device)
    return agent


def create_cnn_convlstm_agent(input_channels: int = 12,
                              input_height: int = 64,
                              input_width: int = 64,
                              hidden_size: int = 256,
                              convlstm_layers: int = 2,
                              device: str = "cpu") -> CNNConvLSTMAgent:
    """Create a CNN+ConvLSTM agent for spatiotemporal feature extraction.

    ConvLSTM captures both spatial and temporal patterns in game state frames,
    making it ideal for understanding dynamic battle situations where the
    spatial arrangement of units evolves over time.

    Args:
        input_channels: Number of input channels (state features per grid cell).
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        convlstm_layers: Number of ConvLSTM layers.
        device: Device to place the network on.

    Returns:
        CNNConvLSTMAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_convlstm",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=convlstm_layers,
    )
    agent = CNNConvLSTMAgent(arch).to(device)
    return agent


def create_cnn_crnn_agent(input_channels: int = 12,
                          input_height: int = 64,
                          input_width: int = 64,
                          hidden_size: int = 256,
                          crnn_layers: int = 2,
                          device: str = "cpu") -> CNNCRNNAgent:
    """Create a CNN+CRNN (Convolutional RNN) agent.

    CRNN applies convolutions along the sequence dimension, capturing
    local temporal patterns more efficiently than standard RNNs. Useful for
    identifying short-term tactical patterns in unit movements and engagements.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        crnn_layers: Number of CRNN layers.
        device: Device to place the network on.

    Returns:
        CNNCRNNAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_crnn",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        lstm_layers=crnn_layers,
    )
    agent = CNNCRNNAgent(arch).to(device)
    return agent


def create_ensemble_agent(agent_list: list, weights: Optional[list] = None) -> EnsembleAgent:
    """Create an ensemble of agents for improved prediction stability.

    Combines predictions from multiple agents using weighted averaging,
    which typically produces more stable and generalizable predictions
    than any single agent.

    Args:
        agent_list: List of trained agent models.
        weights: Optional prediction weights (defaults to uniform).

    Returns:
        EnsembleAgent instance.
    """
    return EnsembleAgent(agent_list, weights)


def create_resnet_variant(input_channels: int = 12,
                          input_height: int = 64,
                          input_width: int = 64,
                          hidden_size: int = 256,
                          num_blocks: int = 3,
                          device: str = "cpu") -> CNNResNetAgent:
    """Create a CNN+ResNet with custom depth.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        num_blocks: Number of residual blocks.
        device: Device to place the network on.

    Returns:
        CNNResNetAgent instance.
    """
    arch = AgentArchitecture(
        name="cnn_resnet",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
        residual=True,
    )
    agent = CNNResNetAgent(arch, num_blocks=num_blocks).to(device)
    return agent


def create_dueling_agent(input_channels: int = 12,
                         input_height: int = 64,
                         input_width: int = 64,
                         hidden_size: int = 256,
                         device: str = "cpu") -> DuelingAgent:
    """Create a Dueling DQN-style agent for action-value estimation.

    Separates state value and advantage streams to better estimate
    which actions are valuable regardless of the current state value.

    Args:
        input_channels: Number of input channels.
        input_height: Input tensor height.
        input_width: Input tensor width.
        hidden_size: Hidden layer size.
        device: Device to place the network on.

    Returns:
        DuelingAgent instance.
    """
    arch = AgentArchitecture(
        name="dueling_cnn",
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        hidden_size=hidden_size,
    )
    agent = DuelingAgent(arch).to(device)
    return agent


# Architecture registry for easy lookup by name
ARCHITECTURE_REGISTRY = {
    "cnn_lstm": create_cnn_lstm_agent,
    "cnn_mlp": create_cnn_mlp_agent,
    "cnn_resnet": create_cnn_resnet_agent,
    "cnn_transformer": create_cnn_transformer_agent,
    "cnn_cnn_mlp": create_cnn_cnn_mlp_agent,
    "cnn_gru": create_cnn_gru_agent,
    "cnn_lstm_attention": create_cnn_lstm_attention_agent,
    "cnn_transformer_lstm": create_cnn_transformer_lstm_agent,
    "cnn_convlstm": create_cnn_convlstm_agent,
    "cnn_crnn": create_cnn_crnn_agent,
    "dueling": create_dueling_agent,
}


def get_architecture(name: str):
    """Get an architecture creation function by name.

    Args:
        name: Architecture name from ARCHITECTURE_REGISTRY.

    Returns:
        Architecture creation function.

    Raises:
        ValueError: If architecture name not found.
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY.keys()))
        raise ValueError(f"Unknown architecture '{name}'. Available: {available}")
    return ARCHITECTURE_REGISTRY[name]
