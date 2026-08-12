"""Tests for the evolutionary agent.

Tests:
- Forward pass
- Action selection (greedy and exploration)
- Weight management
- Checkpoint save/load
- Epsilon decay
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models import EvolutionaryAgent, AgentConfig


class TestEvolutionaryAgent:
    """Tests for the EvolutionaryAgent class."""

    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm"),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)
        card_logits, placement = agent.forward(state)

        assert card_logits.shape == (5,)  # 4 cards + pass
        assert placement.shape == (2,)
        assert np.all(np.isfinite(card_logits))
        assert np.all(np.isfinite(placement))

    def test_forward_pass_mlp(self):
        """Test CNN+MLP forward pass."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_mlp"),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)
        card_logits, placement = agent.forward(state)

        assert card_logits.shape == (5,)
        assert placement.shape == (2,)

    def test_select_action_greedy(self):
        """Test greedy action selection (epsilon=0)."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm", epsilon=0.0),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)

        actions = []
        for _ in range(10):
            action = agent.select_action(state)
            actions.append(action)
            assert not action["is_exploration"]

        # With epsilon=0, actions should be deterministic
        assert all(a["card_idx"] == actions[0]["card_idx"] for a in actions)

    def test_select_action_exploration(self):
        """Test exploratory action selection (epsilon=1.0)."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm", epsilon=1.0),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)

        actions = []
        for _ in range(100):
            action = agent.select_action(state)
            actions.append(action)

        # With epsilon=1.0, should see exploration
        exploration_count = sum(1 for a in actions if a["is_exploration"])
        assert exploration_count > 0

    def test_select_action_pass(self):
        """Test that pass action is possible."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm", epsilon=1.0),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)

        actions = []
        for _ in range(500):
            action = agent.select_action(state)
            actions.append(action)

        pass_count = sum(1 for a in actions if a["card_idx"] == -1)
        assert pass_count > 0

    def test_get_weights(self):
        """Test weight extraction."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm"),
            seed=42,
        )

        weights = agent.get_weights()
        assert isinstance(weights, np.ndarray)
        assert weights.shape[0] > 0

    def test_set_weights(self):
        """Test weight setting."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm"),
            seed=42,
        )

        original_weights = agent.get_weights()
        new_weights = np.random.randn(original_weights.shape[0]) * 0.1

        agent.set_weights(new_weights)
        current_weights = agent.get_weights()

        assert np.allclose(current_weights, new_weights)
        assert not np.allclose(current_weights, original_weights)

    def test_epsilon_decay(self):
        """Test epsilon decay."""
        agent = EvolutionaryAgent(
            config=AgentConfig(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01),
            seed=42,
        )

        initial_epsilon = agent.get_exploration_rate()
        assert initial_epsilon == 1.0

        for _ in range(50):
            agent.decay_epsilon()

        current_epsilon = agent.get_exploration_rate()
        assert current_epsilon < initial_epsilon
        assert current_epsilon >= 0.01  # Should not go below epsilon_min

    def test_epsilon_min_clamp(self):
        """Test epsilon doesn't go below minimum."""
        agent = EvolutionaryAgent(
            config=AgentConfig(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.05),
            seed=42,
        )

        for _ in range(100):
            agent.decay_epsilon()

        assert agent.get_exploration_rate() >= 0.05

    def test_save_load_checkpoint(self):
        """Test checkpoint save and load."""
        agent = EvolutionaryAgent(
            config=AgentConfig(epsilon=0.3),
            seed=42,
        )

        # Set some state
        agent.epsilon = 0.5
        agent.total_actions = 1000
        original_weights = agent.get_weights()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "agent.pt")
            agent.save_checkpoint(path)

            # Create new agent and load
            new_agent = EvolutionaryAgent(
                config=AgentConfig(epsilon=0.0),
                seed=99,
            )
            new_agent.load_checkpoint(path)

            assert np.allclose(new_agent.get_weights(), original_weights)
            assert new_agent.epsilon == 0.5
            assert new_agent.total_actions == 1000

    def test_reset_lstm(self):
        """Test LSTM reset."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm"),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)
        _ = agent.forward(state)

        # Reset should clear hidden state
        agent.reset()
        assert agent.network.hidden_state is None
        assert agent.network.cell_state is None

    def test_agent_repr(self):
        """Test agent string representation."""
        agent = EvolutionaryAgent(
            config=AgentConfig(architecture="cnn_lstm", epsilon=0.1),
            seed=42,
        )

        repr_str = repr(agent)
        assert "cnn_lstm" in repr_str
        assert "epsilon=" in repr_str
        assert "weights_shape=" in repr_str


class TestActionSelection:
    """Tests for action selection edge cases."""

    def test_action_with_valid_cards(self):
        """Test action selection with card validity mask."""
        agent = EvolutionaryAgent(
            config=AgentConfig(epsilon=0.0),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)
        valid_cards = np.array([True, True, False, False])

        action = agent.select_action(state, valid_cards=valid_cards)
        assert action["card_idx"] in [0, 1, -1]

    def test_action_consistency(self):
        """Test that greedy actions are consistent."""
        agent = EvolutionaryAgent(
            config=AgentConfig(epsilon=0.0),
            seed=42,
        )

        state = np.random.randn(12, 64, 64).astype(np.float32)

        actions = [agent.select_action(state) for _ in range(10)]
        first_action = actions[0]

        for action in actions[1:]:
            assert action["card_idx"] == first_action["card_idx"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
