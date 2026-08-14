"""Tests for the tournament system.

Tests:
- TournamentFormat enum
- HeadToHeadRecord
- AgentTournamentStats
- TournamentRunner (round-robin, single elimination, double elimination, league)
- TournamentResult
- TournamentEvolutionStrategy
- FitnessEvaluator.run_tournament
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.train import (
    TournamentFormat,
    TournamentResult,
    TournamentRunner,
    AgentTournamentStats,
    HeadToHeadRecord,
    TournamentBracket,
    FitnessEvaluator,
)
from src.models import TournamentEvolutionStrategy


# =============================================================================
# HeadToHeadRecord Tests
# =============================================================================


class TestHeadToHeadRecord:
    """Tests for HeadToHeadRecord."""

    def test_matches_count(self):
        """Test matches property returns correct total."""
        record = HeadToHeadRecord(
            agent1_id="a1", agent2_id="a2",
            agent1_wins=3, agent2_wins=2, draws=1,
        )
        assert record.matches == 6

    def test_win_rates(self):
        """Test win rate calculations."""
        record = HeadToHeadRecord(
            agent1_id="a1", agent2_id="a2",
            agent1_wins=3, agent2_wins=1, draws=2,
        )
        assert record.agent1_win_rate == pytest.approx(3 / 6)
        assert record.agent2_win_rate == pytest.approx(1 / 6)

    def test_tower_differential(self):
        """Test tower differential calculation."""
        record = HeadToHeadRecord(
            agent1_id="a1", agent2_id="a2",
            agent1_towers=5, agent2_towers=3,
        )
        assert record.tower_differential == 2

    def test_zero_matches_win_rate(self):
        """Test win rate is 0 when no matches played."""
        record = HeadToHeadRecord(agent1_id="a1", agent2_id="a2")
        assert record.agent1_win_rate == 0.0
        assert record.agent2_win_rate == 0.0


# =============================================================================
# AgentTournamentStats Tests
# =============================================================================


class TestAgentTournamentStats:
    """Tests for AgentTournamentStats."""

    def test_matches_count(self):
        """Test matches property."""
        stats = AgentTournamentStats(agent_id="a1", wins=3, draws=2, losses=1)
        assert stats.matches == 6

    def test_win_rate(self):
        """Test win rate calculation."""
        stats = AgentTournamentStats(agent_id="a1", wins=5, draws=0, losses=5)
        assert stats.win_rate == pytest.approx(0.5)

    def test_zero_matches_win_rate(self):
        """Test win rate is 0 when no matches."""
        stats = AgentTournamentStats(agent_id="a1")
        assert stats.win_rate == 0.0

    def test_tower_differential(self):
        """Test tower differential."""
        stats = AgentTournamentStats(agent_id="a1", towers_destroyed=10, losses=3)
        assert stats.tower_differential == 7

    def test_composite_score(self):
        """Score is points-per-match plus a tower term.

        Normalising by matches keeps Swiss pairings comparable when agents
        play unequal numbers of games. Match duration is deliberately not a
        term: the old formula added 0.01 * avg_duration, and with durations in
        the hundreds of ticks that outweighed every win.
        """
        stats = AgentTournamentStats(agent_id="a1", wins=5, draws=2, losses=3,
                                     towers_destroyed=20, total_duration=600)
        score = stats.compute_composite_score()
        expected = (5 * 1.0 + 2 * 0.5) / 10 + 0.1 * (20 / 10)
        assert score == pytest.approx(expected, rel=1e-5)

    def test_composite_score_ignores_match_duration(self):
        slow = AgentTournamentStats(agent_id="slow", wins=1, losses=1,
                                    total_duration=100_000)
        quick = AgentTournamentStats(agent_id="quick", wins=1, losses=1,
                                     total_duration=10)
        assert slow.compute_composite_score() == quick.compute_composite_score()

    def test_composite_score_rewards_winning(self):
        winner = AgentTournamentStats(agent_id="w", wins=8, losses=2)
        loser = AgentTournamentStats(agent_id="l", wins=2, losses=8)
        assert winner.compute_composite_score() > loser.compute_composite_score()

    def test_update_elo_draw_between_equals_is_neutral(self):
        """A draw between equally rated agents should not move ratings."""
        stats = AgentTournamentStats(agent_id="a1", elo_rating=1500.0)
        stats.update_elo(opponent_elo=1500.0, actual_score=0.5, k_factor=32.0)
        assert stats.elo_rating == pytest.approx(1500.0, abs=0.1)

    def test_update_elo_win(self):
        """Beating a higher-rated opponent must raise the rating.

        The old implementation derived "actual" from the ratings
        (actual = 1 - expected) and ignored the result argument entirely, so
        the update reduced to k*(1 - 2*expected) and moved ratings without
        ever consulting who won.
        """
        stats = AgentTournamentStats(agent_id="a1", elo_rating=1500.0)
        stats.update_elo(opponent_elo=1600.0, actual_score=1.0, k_factor=32.0)
        assert stats.elo_rating > 1500.0

    def test_update_elo_loss(self):
        stats = AgentTournamentStats(agent_id="a1", elo_rating=1500.0)
        stats.update_elo(opponent_elo=1400.0, actual_score=0.0, k_factor=32.0)
        assert stats.elo_rating < 1500.0

    def test_update_elo_reflects_the_result_not_just_the_ratings(self):
        """Same matchup, opposite results, opposite rating movement."""
        winner = AgentTournamentStats(agent_id="w", elo_rating=1500.0)
        loser = AgentTournamentStats(agent_id="l", elo_rating=1500.0)
        winner.update_elo(opponent_elo=1500.0, actual_score=1.0)
        loser.update_elo(opponent_elo=1500.0, actual_score=0.0)
        assert winner.elo_rating > loser.elo_rating

    def test_beating_a_stronger_opponent_gains_more(self):
        vs_strong = AgentTournamentStats(agent_id="a", elo_rating=1500.0)
        vs_weak = AgentTournamentStats(agent_id="b", elo_rating=1500.0)
        vs_strong.update_elo(opponent_elo=1800.0, actual_score=1.0)
        vs_weak.update_elo(opponent_elo=1200.0, actual_score=1.0)
        assert vs_strong.elo_rating > vs_weak.elo_rating


# =============================================================================
# TournamentResult Tests
# =============================================================================


class TestTournamentResult:
    """Tests for TournamentResult."""

    def test_ranking(self):
        """Test get_ranking method."""
        result = TournamentResult(
            rankings=[("a1", 10.0), ("a2", 8.0), ("a3", 6.0)],
        )
        assert result.get_ranking("a1") == 1
        assert result.get_ranking("a2") == 2
        assert result.get_ranking("a3") == 3
        assert result.get_ranking("unknown") == 4

    def test_win_rate(self):
        """Test get_win_rate method."""
        stats = AgentTournamentStats(agent_id="a1", wins=5, draws=0, losses=5)
        result = TournamentResult(agent_stats={"a1": stats})
        assert result.get_win_rate("a1") == pytest.approx(0.5)

    def test_h2h_record(self):
        """Test get_h2h_record method."""
        h2h = HeadToHeadRecord(agent1_id="a1", agent2_id="a2", agent1_wins=3, agent2_wins=1)
        result = TournamentResult(h2h_records={("a1", "a2"): h2h})
        assert result.get_h2h_record("a1", "a2") is h2h
        assert result.get_h2h_record("a2", "a1") is h2h  # Order doesn't matter
        assert result.get_h2h_record("a1", "a3") is None

    def test_summary(self):
        """Test summary generation."""
        stats = AgentTournamentStats(agent_id="a1", wins=5, draws=2, losses=3,
                                      towers_destroyed=20, elo_rating=1600.0)
        result = TournamentResult(
            rankings=[("a1", 7.5)],
            agent_stats={"a1": stats},
            total_matches=10,
            generation=5,
        )
        summary = result.summary()
        assert "Tournament Results" in summary
        assert "Generation 5" in summary
        assert "a1" in summary
        assert "10" in summary


# =============================================================================
# TournamentRunner Tests (without actual simulation)
# =============================================================================


class TestTournamentRunner:
    """Tests for TournamentRunner."""

    def test_round_robin_format(self):
        """Test round-robin format enum value."""
        assert TournamentFormat.ROUND_ROBIN is not None

    def test_single_elimination_format(self):
        """Test single elimination format enum value."""
        assert TournamentFormat.SINGLE_ELIMINATION is not None

    def test_double_elimination_format(self):
        """Test double elimination format enum value."""
        assert TournamentFormat.DOUBLE_ELIMINATION is not None

    def test_league_format(self):
        """Test league format enum value."""
        assert TournamentFormat.LEAGUE is not None


# =============================================================================
# TournamentEvolutionStrategy Tests
# =============================================================================


class TestTournamentEvolutionStrategy:
    """Tests for TournamentEvolutionStrategy."""

    def test_init_default(self):
        """Test default initialization."""
        strategy = TournamentEvolutionStrategy()
        assert strategy.tournament_format == "ROUND_ROBIN"
        assert strategy.matches_per_pair == 4
        assert strategy.elite_fraction == 0.1
        assert strategy.crossover_rate == 0.7
        assert strategy.mutation_rate == 0.05
        assert strategy.mutation_std == 0.1

    def test_evolve_without_evaluator(self):
        """Test evolve without evaluator (uses fitness fallback)."""
        strategy = TournamentEvolutionStrategy(seed=42)

        population = [np.random.randn(100) for _ in range(10)]
        fitnesses = [float(i) for i in range(10)]

        offspring, info = strategy.evolve(
            population=population,
            weights_list=population,
            current_fitnesses=fitnesses,
            generation=0,
        )

        assert len(offspring) == 10
        assert "tournament_rankings" in info
        assert "elite_indices" in info
        assert "elo_ratings" in info
        assert len(info["elite_indices"]) > 0

    def test_evolve_preserves_elites(self):
        """Test that elites are preserved in offspring."""
        strategy = TournamentEvolutionStrategy(
            elite_fraction=0.2,  # 2 elites
            seed=42,
        )

        population = [np.random.randn(100) for _ in range(10)]
        fitnesses = [float(i) for i in range(10)]

        offspring, info = strategy.evolve(
            population=population,
            weights_list=population,
            current_fitnesses=fitnesses,
            generation=0,
        )

        elite_count = max(1, int(10 * 0.2))
        assert len(info["elite_indices"]) == elite_count

    def test_tournament_summary(self):
        """Test tournament summary generation."""
        strategy = TournamentEvolutionStrategy(seed=42)

        # Simulate some tournament history
        for gen in range(5):
            info = {
                "generation": gen,
                "rankings": [(f"agent_{i}", float(10 - i)) for i in range(10)],
                "elite_indices": [0, 1],
                "elo_ratings": {f"agent_{i}": 1500.0 + i for i in range(10)},
            }
            strategy.tournament_history.append(info)

        summary = strategy.get_tournament_summary(last_n=3)
        assert "generations" in summary
        assert "elite_indices" in summary
        assert "top_score_history" in summary
        assert len(summary["generations"]) == 3

    def test_offspring_different_from_parents(self):
        """Test that offspring are different from parents (mutation)."""
        strategy = TournamentEvolutionStrategy(
            mutation_rate=1.0,  # High mutation for testing
            mutation_std=0.5,
            seed=42,
        )

        population = [np.ones(100) * i for i in range(10)]
        fitnesses = [float(i) for i in range(10)]

        offspring, _ = strategy.evolve(
            population=population,
            weights_list=population,
            current_fitnesses=fitnesses,
            generation=0,
        )

        # Offspring should be different from parents due to mutation
        for i, child in enumerate(offspring):
            # At least some children should differ from all parents
            is_different = False
            for j, parent in enumerate(population):
                if not np.allclose(child, parent, atol=0.01):
                    is_different = True
                    break
            assert is_different, f"Offspring {i} is identical to all parents"


# =============================================================================
# Integration Tests (lightweight, no actual simulation)
# =============================================================================


class TestTournamentIntegration:
    """Integration tests for the tournament system."""

    def test_tournament_result_flow(self):
        """Test creating and using a TournamentResult."""
        stats_a = AgentTournamentStats(agent_id="a", wins=5, draws=1, losses=4,
                                        towers_destroyed=15, elo_rating=1550.0)
        stats_b = AgentTournamentStats(agent_id="b", wins=4, draws=2, losses=4,
                                        towers_destroyed=12, elo_rating=1480.0)

        h2h = HeadToHeadRecord(agent1_id="a", agent2_id="b",
                                agent1_wins=3, agent2_wins=2, draws=2)

        result = TournamentResult(
            rankings=[("a", 6.0), ("b", 5.0)],
            agent_stats={"a": stats_a, "b": stats_b},
            h2h_records={("a", "b"): h2h},
            wins={"a": 5, "b": 4},
            draws={"a": 1, "b": 2},
            losses={"a": 4, "b": 4},
            total_matches=10,
            elo_ratings={"a": 1550.0, "b": 1480.0},
            generation=1,
        )

        # Test all accessors
        assert result.get_ranking("a") == 1
        assert result.get_ranking("b") == 2
        assert result.get_win_rate("a") == pytest.approx(5 / 10)
        assert result.get_h2h_record("a", "b") is h2h
        assert result.summary() is not None

    def test_bracket_creation(self):
        """Test TournamentBracket creation."""
        bracket = TournamentBracket(
            format=TournamentFormat.SINGLE_ELIMINATION,
            rounds=[(("a", "b"), ("c", "d"))],
            winners=["a"],
            loser_bracket=["b", "c"],
        )
        assert bracket.format == TournamentFormat.SINGLE_ELIMINATION
        assert len(bracket.rounds) == 1
        assert bracket.winners == ["a"]

    def test_elo_history_tracking(self):
        """Test ELO history tracking in TournamentEvolutionStrategy."""
        strategy = TournamentEvolutionStrategy(seed=42)

        # Simulate tournament results
        for gen in range(3):
            for i in range(5):
                aid = f"agent_{i}"
                if aid not in strategy.elo_history:
                    strategy.elo_history[aid] = []
                strategy.elo_history[aid].append(1500.0 + gen * 10 + i)

        assert "agent_0" in strategy.elo_history
        assert len(strategy.elo_history["agent_0"]) == 3
        assert strategy.elo_history["agent_0"] == [1500.0, 1510.0, 1520.0]
