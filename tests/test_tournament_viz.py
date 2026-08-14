"""Tests for tournament visualization utilities.

Tests:
- ASCII bracket rendering
- ELO progression chart creation
- Win rate chart creation
- H2H matrix chart creation
- Tournament summary computation
- JSON serialization/deserialization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.viz.tournament_viz import (
    render_bracket_ascii,
    create_elo_progression_chart,
    create_win_rate_chart,
    create_h2h_matrix_chart,
    compute_tournament_summary,
    TournamentSummary,
    tournament_result_to_dict,
    tournament_result_from_dict,
)
from src.train import (
    TournamentResult,
    AgentTournamentStats,
    HeadToHeadRecord,
    TournamentBracket,
    TournamentFormat,
)


# =============================================================================
# ASCII Bracket Tests
# =============================================================================


class TestRenderBracketAscii:
    """Tests for ASCII bracket rendering."""

    def test_render_round_robin_bracket(self):
        """Test rendering a round-robin bracket."""
        bracket = TournamentBracket(format=TournamentFormat.ROUND_ROBIN)
        result = render_bracket_ascii(bracket)
        assert "ROUND_ROBIN" in result
        assert "=" in result

    def test_render_single_elimination_bracket(self):
        """Test rendering a single elimination bracket."""
        bracket = TournamentBracket(
            format=TournamentFormat.SINGLE_ELIMINATION,
            rounds=[[("agent_0", "agent_1"), ("agent_2", "agent_3")]],
            winners=["agent_0"],
            loser_bracket=["agent_1", "agent_2"],
        )
        result = render_bracket_ascii(bracket)
        assert "SINGLE_ELIMINATION" in result
        assert "agent_0" in result
        assert "agent_1" in result
        assert "Round 1" in result
        assert "Winners" in result
        assert "Loser Bracket" in result

    def test_render_with_bye(self):
        """Test rendering with a bye."""
        bracket = TournamentBracket(
            format=TournamentFormat.SINGLE_ELIMINATION,
            rounds=[(("agent_0", None),)],
            winners=["agent_0"],
        )
        result = render_bracket_ascii(bracket)
        assert "BYE" in result

    def test_render_none_bracket(self):
        """Test rendering None bracket."""
        result = render_bracket_ascii(None)
        assert "No bracket data available" in result


# =============================================================================
# Chart Creation Tests
# =============================================================================


class TestChartCreation:
    """Tests for chart creation functions."""

    def test_elo_chart_returns_figure(self):
        """Test ELO chart creation returns a figure."""
        elo_history = {
            "agent_0": [1500.0, 1520.0, 1540.0],
            "agent_1": [1500.0, 1480.0, 1460.0],
        }
        fig = create_elo_progression_chart(elo_history)
        # Plotly may not be available, so check for None or figure
        if fig is not None:
            assert hasattr(fig, 'data')
            assert len(fig.data) == 2

    def test_elo_chart_empty(self):
        """Test ELO chart with empty history."""
        fig = create_elo_progression_chart({})
        # May return None or empty figure - just check it doesn't crash
        assert fig is None or hasattr(fig, 'data')

    def test_win_rate_chart_returns_figure(self):
        """Test win rate chart creation."""
        rankings = [("agent_0", 10.0), ("agent_1", 8.0), ("agent_2", 6.0)]
        agent_stats = {
            "agent_0": AgentTournamentStats(agent_id="agent_0", wins=5, draws=0, losses=5),
            "agent_1": AgentTournamentStats(agent_id="agent_1", wins=4, draws=2, losses=4),
            "agent_2": AgentTournamentStats(agent_id="agent_2", wins=3, draws=1, losses=6),
        }
        fig = create_win_rate_chart(rankings, agent_stats)
        if fig is not None:
            assert hasattr(fig, 'data')

    def test_h2h_matrix_chart(self):
        """Test H2H matrix chart creation."""
        h2h_records = {
            ("agent_0", "agent_1"): HeadToHeadRecord(
                agent1_id="agent_0", agent2_id="agent_1",
                agent1_wins=3, agent2_wins=1, draws=2,
            ),
            ("agent_0", "agent_2"): HeadToHeadRecord(
                agent1_id="agent_0", agent2_id="agent_2",
                agent1_wins=2, agent2_wins=2, draws=2,
            ),
            ("agent_1", "agent_2"): HeadToHeadRecord(
                agent1_id="agent_1", agent2_id="agent_2",
                agent1_wins=4, agent2_wins=0, draws=2,
            ),
        }
        agent_ids = ["agent_0", "agent_1", "agent_2"]
        fig = create_h2h_matrix_chart(h2h_records, agent_ids)
        if fig is not None:
            assert hasattr(fig, 'data')


# =============================================================================
# Tournament Summary Tests
# =============================================================================


class TestTournamentSummary:
    """Tests for tournament summary computation."""

    def test_compute_summary(self):
        """Test computing tournament summary."""
        stats_a = AgentTournamentStats(agent_id="agent_0", wins=5, draws=1, losses=4,
                                        towers_destroyed=15, elo_rating=1550.0)
        stats_b = AgentTournamentStats(agent_id="agent_1", wins=4, draws=2, losses=4,
                                        towers_destroyed=12, elo_rating=1480.0)
        stats_c = AgentTournamentStats(agent_id="agent_2", wins=3, draws=3, losses=4,
                                        towers_destroyed=10, elo_rating=1470.0)

        result = TournamentResult(
            rankings=[("agent_0", 7.5), ("agent_1", 6.0), ("agent_2", 5.0)],
            agent_stats={"agent_0": stats_a, "agent_1": stats_b, "agent_2": stats_c},
            h2h_records={
                ("agent_0", "agent_1"): HeadToHeadRecord(
                    agent1_id="agent_0", agent2_id="agent_1",
                    agent1_wins=3, agent2_wins=2, draws=2,
                ),
            },
            total_matches=10,
            elo_ratings={"agent_0": 1550.0, "agent_1": 1480.0, "agent_2": 1470.0},
            generation=5,
        )

        summary = compute_tournament_summary(result)

        assert summary.total_agents == 3
        assert summary.total_matches == 10
        assert summary.top_agent == "agent_0"
        assert summary.top_elo == 1550.0
        assert summary.elo_spread == 80.0  # 1550 - 1470
        assert 0.0 <= summary.competitiveness <= 1.0

    def test_summary_to_dict(self):
        """Test converting summary to dictionary."""
        summary = TournamentSummary(
            total_agents=10,
            total_matches=100,
            avg_win_rate=0.5,
            elo_spread=100.0,
            top_agent="agent_0",
            top_elo=1600.0,
            competitiveness=0.8,
        )
        d = summary.to_dict()
        assert d["total_agents"] == 10
        assert d["total_matches"] == 100
        assert d["top_agent"] == "agent_0"
        assert d["top_elo"] == 1600.0

    def test_competitiveness_levels(self):
        """Test competitiveness calculation at different spreads."""
        # Create results with different ELO spreads
        for elo_spread, expected_level in [(0, ">0.8"), (50, ">0.5"), (200, ">0.2")]:
            base_elo = 1500.0
            stats = {}
            rankings = []
            elos = {}
            for i in range(5):
                elo = base_elo - (i * elo_spread / 4)
                stats[f"agent_{i}"] = AgentTournamentStats(
                    agent_id=f"agent_{i}", elo_rating=elo,
                    wins=5 - i, losses=i, draws=0,
                )
                rankings.append((f"agent_{i}", elo))
                elos[f"agent_{i}"] = elo

            result = TournamentResult(
                rankings=rankings,
                agent_stats=stats,
                elo_ratings=elos,
            )
            summary = compute_tournament_summary(result)

            if elo_spread == 0:
                assert summary.competitiveness > 0.8, f"Expected competitive for spread 0"
            elif elo_spread <= 100:
                assert summary.competitiveness > 0.5, f"Expected competitive for spread {elo_spread}"


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJsonSerialization:
    """Tests for JSON serialization of tournament results."""

    def test_roundtrip_serialization(self):
        """Test serializing and deserializing a tournament result."""
        stats_a = AgentTournamentStats(agent_id="agent_0", wins=5, draws=1, losses=4,
                                        towers_destroyed=15, total_duration=600,
                                        elo_rating=1550.0)
        stats_b = AgentTournamentStats(agent_id="agent_1", wins=4, draws=2, losses=4,
                                        towers_destroyed=12, total_duration=580,
                                        elo_rating=1480.0)

        h2h = HeadToHeadRecord(
            agent1_id="agent_0", agent2_id="agent_1",
            agent1_wins=3, agent2_wins=2, draws=2,
            agent1_towers=10, agent2_towers=8,
            agent1_duration=300, agent2_duration=290,
        )

        original = TournamentResult(
            rankings=[("agent_0", 7.5), ("agent_1", 6.0)],
            agent_stats={"agent_0": stats_a, "agent_1": stats_b},
            h2h_records={("agent_0", "agent_1"): h2h},
            wins={"agent_0": 5, "agent_1": 4},
            draws={"agent_0": 1, "agent_1": 2},
            losses={"agent_0": 4, "agent_1": 4},
            total_matches=10,
            elo_ratings={"agent_0": 1550.0, "agent_1": 1480.0},
            generation=5,
        )

        # Serialize
        data = tournament_result_to_dict(original)
        assert "generation" in data
        assert "rankings" in data
        assert "elo_ratings" in data
        assert "agent_stats" in data
        assert "h2h_records" in data
        assert "summary" in data

        # Deserialize
        restored = tournament_result_from_dict(data)
        assert restored.generation == original.generation
        assert restored.total_matches == original.total_matches
        assert len(restored.rankings) == len(original.rankings)
        assert restored.rankings[0][0] == original.rankings[0][0]

    def test_empty_result_serialization(self):
        """Test serializing an empty tournament result."""
        original = TournamentResult(generation=0)
        data = tournament_result_to_dict(original)
        restored = tournament_result_from_dict(data)
        assert restored.generation == 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_agent_tournament(self):
        """Test summary with single agent."""
        stats = AgentTournamentStats(agent_id="agent_0", wins=10, elo_rating=1500.0)
        result = TournamentResult(
            rankings=[("agent_0", 10.0)],
            agent_stats={"agent_0": stats},
            elo_ratings={"agent_0": 1500.0},
        )
        summary = compute_tournament_summary(result)
        assert summary.total_agents == 1
        assert summary.elo_spread == 0.0

    def test_no_h2h_records(self):
        """Test H2H chart with no records."""
        fig = create_h2h_matrix_chart({}, ["agent_0", "agent_1"])
        if fig is not None:
            assert hasattr(fig, 'data')

    def test_elo_chart_with_many_agents(self):
        """Test ELO chart with many agents (should limit display)."""
        elo_history = {f"agent_{i}": [1500.0 + i] for i in range(20)}
        fig = create_elo_progression_chart(elo_history)
        if fig is not None:
            # Should only display top 10
            assert len(fig.data) <= 10
