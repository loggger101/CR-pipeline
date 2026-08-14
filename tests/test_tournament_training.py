"""Tests for tournament matchmaking as the primary training process.

Agents are trained by playing each other: fitness is a competitor's standing in
a Swiss tournament against the rest of the population plus the hall of fame.

The subtle risk with self-referential fitness is that a rising mean proves
nothing -- every agent can improve *relative to the field* while the field goes
nowhere, or cycles. The absolute checks at the bottom guard against that.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim.parallel_runner import ParallelRunner, _run_head_to_head
from src.models.policy import DEFAULT_POLICY_SPEC
from src.train.evaluator import (
    DEFAULT_ELO, AgentTournamentStats, TournamentFormat, TournamentRunner,
)
from src.train.trainer import EvolutionTrainer, TrainingConfig


def _genome(seed: int) -> np.ndarray:
    return DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(seed))


@pytest.fixture(scope="module")
def runner():
    pool = ParallelRunner(num_workers=4)
    yield pool
    pool.shutdown()


class TestSwissPairing:
    """Pairing arithmetic and structure, no matches needed."""

    @pytest.mark.parametrize("n,expected", [
        (2, 1), (4, 2), (8, 3), (16, 4), (64, 6), (200, 8),
    ])
    def test_recommended_rounds_is_log2(self, n, expected):
        assert TournamentRunner.recommended_rounds(n) == expected

    def test_swiss_is_far_cheaper_than_round_robin(self):
        """This is what makes agent-vs-agent viable as the main loop."""
        n = 200
        rounds = TournamentRunner.recommended_rounds(n)
        swiss = n * rounds // 2
        round_robin = n * (n - 1) // 2
        assert swiss < round_robin / 20

    def test_pairing_covers_everyone_once_per_round(self):
        order = list(range(8))
        pairings, bye = TournamentRunner._swiss_pair(order, set())
        assert bye is None
        assert len(pairings) == 4
        seen = [i for pair in pairings for i in pair]
        assert sorted(seen) == order

    def test_odd_field_produces_exactly_one_bye(self):
        order = list(range(7))
        pairings, bye = TournamentRunner._swiss_pair(order, set())
        assert bye is not None
        assert len(pairings) == 3
        seen = [i for pair in pairings for i in pair] + [bye]
        assert sorted(seen) == order

    def test_pairing_avoids_rematches_when_possible(self):
        order = [0, 1, 2, 3]
        played = {(0, 1), (2, 3)}
        pairings, _bye = TournamentRunner._swiss_pair(order, played)
        for a, b in pairings:
            assert (min(a, b), max(a, b)) not in played

    def test_pairing_still_completes_when_all_are_rematches(self):
        """Falling back to a rematch beats failing to pair at all."""
        order = [0, 1]
        pairings, bye = TournamentRunner._swiss_pair(order, {(0, 1)})
        assert pairings == [(0, 1)]
        assert bye is None


class TestSwissTournament:
    """End-to-end Swiss runs over real genomes."""

    def test_every_agent_plays_every_round(self, runner):
        ids = [f"agent_{i}" for i in range(8)]
        genomes = [_genome(300 + i) for i in range(8)]
        tr = TournamentRunner(runner, seed=5)

        result = tr.run_swiss(ids, genomes, matches_per_pair=2, rounds=3)

        for agent_id in ids:
            assert result.agent_stats[agent_id].matches == 6  # 3 rounds x 2

    def test_produces_a_full_ranking(self, runner):
        ids = [f"agent_{i}" for i in range(8)]
        genomes = [_genome(300 + i) for i in range(8)]
        tr = TournamentRunner(runner, seed=5)

        result = tr.run_swiss(ids, genomes, matches_per_pair=2, rounds=3)

        assert len(result.rankings) == 8
        assert {aid for aid, _ in result.rankings} == set(ids)
        scores = [score for _, score in result.rankings]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_tracks_skill_not_noise(self, runner):
        """Two independent tournaments over the same field must broadly agree.

        A near-zero correlation would mean the standings are chance, and
        selection on them would be sorting noise.
        """
        ids = [f"agent_{i}" for i in range(12)]
        genomes = [_genome(500 + i) for i in range(12)]

        first = TournamentRunner(runner, seed=1).run_swiss(
            ids, genomes, matches_per_pair=2)
        second = TournamentRunner(runner, seed=99).run_swiss(
            ids, genomes, matches_per_pair=2)

        pos1 = {aid: i for i, (aid, _) in enumerate(first.rankings)}
        pos2 = {aid: i for i, (aid, _) in enumerate(second.rankings)}
        x = np.array([pos1[a] for a in ids], dtype=float)
        y = np.array([pos2[a] for a in ids], dtype=float)

        assert np.corrcoef(x, y)[0, 1] > 0.3

    def test_elo_moves_away_from_the_default(self, runner):
        ids = [f"agent_{i}" for i in range(8)]
        genomes = [_genome(300 + i) for i in range(8)]
        tr = TournamentRunner(runner, seed=5)

        result = tr.run_swiss(ids, genomes, matches_per_pair=2, rounds=3)

        ratings = list(result.elo_ratings.values())
        assert max(ratings) > DEFAULT_ELO > min(ratings)

    def test_carried_in_elo_is_respected(self, runner):
        ids = [f"agent_{i}" for i in range(4)]
        genomes = [_genome(700 + i) for i in range(4)]
        tr = TournamentRunner(runner, seed=5)

        seeded = {"agent_0": 2000.0}
        result = tr.run_swiss(ids, genomes, matches_per_pair=2, rounds=1,
                              initial_elo=seeded)

        # One round cannot erase a 500-point head start.
        assert result.elo_ratings["agent_0"] > DEFAULT_ELO + 300

    def test_single_agent_field_is_handled(self, runner):
        tr = TournamentRunner(runner, seed=5)
        result = tr.run_swiss(["solo"], [_genome(1)], matches_per_pair=2)
        assert result.rankings == [("solo", 0.0)]

    def test_swiss_is_the_default_format(self, runner):
        ids = [f"agent_{i}" for i in range(4)]
        genomes = [_genome(800 + i) for i in range(4)]
        tr = TournamentRunner(runner, seed=5)

        result = tr.run_tournament(ids, genomes, matches_per_pair=2)

        # Swiss plays rounds*n/2 matchups; round-robin would play all pairs.
        assert result.total_matches < 4 * 3  # fewer than round-robin's 6 pairs x 2


class TestEloCorrectness:
    """ELO must reflect results, not just prior ratings."""

    def test_pair_update_is_zero_sum_between_equals(self, runner):
        tr = TournamentRunner(runner, seed=0)
        ratings = {"a": DEFAULT_ELO, "b": DEFAULT_ELO}
        tr._update_elo_pair("a", "b", a1_wins=2, a1_draws=0,
                            a2_wins=0, a2_draws=0, elo_ratings=ratings)
        assert ratings["a"] > DEFAULT_ELO
        assert ratings["b"] < DEFAULT_ELO
        assert ratings["a"] + ratings["b"] == pytest.approx(2 * DEFAULT_ELO)

    def test_a_clean_sweep_scores_as_a_full_win(self, runner):
        """The old divisor (wins+draws+1) scored a 2-0 sweep as 0.67."""
        tr = TournamentRunner(runner, seed=0)
        ratings = {"a": DEFAULT_ELO, "b": DEFAULT_ELO}
        tr._update_elo_pair("a", "b", a1_wins=2, a1_draws=0,
                            a2_wins=0, a2_draws=0, elo_ratings=ratings)
        # A full win against an equal opponent moves by k * (1 - 0.5).
        assert ratings["a"] == pytest.approx(DEFAULT_ELO + 16.0, abs=0.01)

    def test_a_split_leaves_ratings_unchanged(self, runner):
        tr = TournamentRunner(runner, seed=0)
        ratings = {"a": DEFAULT_ELO, "b": DEFAULT_ELO}
        tr._update_elo_pair("a", "b", a1_wins=1, a1_draws=0,
                            a2_wins=1, a2_draws=0, elo_ratings=ratings)
        assert ratings["a"] == pytest.approx(DEFAULT_ELO)
        assert ratings["b"] == pytest.approx(DEFAULT_ELO)

    def test_no_games_leaves_ratings_untouched(self, runner):
        tr = TournamentRunner(runner, seed=0)
        ratings = {"a": DEFAULT_ELO, "b": DEFAULT_ELO}
        tr._update_elo_pair("a", "b", 0, 0, 0, 0, ratings)
        assert ratings == {"a": DEFAULT_ELO, "b": DEFAULT_ELO}


class TestTournamentTraining:
    """The trainer's default path."""

    def _config(self, tmpdir, generations=3, **overrides):
        params = dict(
            population_size=8, elite_count=2, max_generations=generations,
            num_workers=2, match_duration="short", runs_dir=tmpdir, seed=3,
            checkpoint_interval=100, curriculum_learning=False,
            diversity_preservation=False, tournament_matches=2,
            hall_of_fame_size=3,
        )
        params.update(overrides)
        return TrainingConfig(**params)

    def test_tournament_mode_is_the_default(self):
        config = TrainingConfig()
        assert config.tournament_mode is True
        assert config.tournament_format == "swiss"

    def test_training_runs_and_records_elo(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(self._config(tmp)) as trainer:
                trainer.train()

                assert trainer.last_tournament is not None
                assert trainer.last_tournament.total_matches > 0
                assert trainer.elo_ratings
                assert any(len(v) > 1 for v in trainer.elo_history.values())

    def test_fitness_comes_from_tournament_standings(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EvolutionTrainer(self._config(tmp, generations=1)) as trainer:
                trainer.population.initialize(seed=3)
                genomes = trainer.population.get_population_weights()
                results = trainer._evaluate_population(genomes, generation=0)

                assert len(results) == len(genomes)
                # Every agent actually played someone.
                assert all(r.wins + r.draws + r.losses > 0 for r in results)
                # Standings separate the field.
                assert len({round(r.fitness, 6) for r in results}) > 1
                assert all("elo" in r.metadata for r in results)

    def test_hall_of_fame_fills_and_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, generations=5, hall_of_fame_size=2)
            with EvolutionTrainer(config) as trainer:
                trainer.train()
                assert len(trainer.hall_of_fame) == 2
                for genome, meta in trainer.hall_of_fame:
                    assert genome.shape == (DEFAULT_POLICY_SPEC.num_params,)
                    assert "generation" in meta

    def test_hall_of_fame_members_compete(self):
        """Champions must actually enter the tournament, not just be stored."""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, generations=3, hall_of_fame_size=2)
            with EvolutionTrainer(config) as trainer:
                trainer.train()
                hof_ids = [k for k in trainer.elo_ratings if k.startswith("hof_")]
                assert hof_ids, "hall of fame never entered a tournament"

    def test_disabling_hall_of_fame_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, generations=2, hall_of_fame_size=0)
            with EvolutionTrainer(config) as trainer:
                trainer.train()
                assert trainer.hall_of_fame == []

    def test_unknown_format_is_rejected_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, tournament_format="knockout")
            with EvolutionTrainer(config) as trainer:
                with pytest.raises(ValueError, match="Unknown tournament_format"):
                    trainer._tournament_format_enum()

    def test_scripted_mode_still_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(tmp, generations=2, tournament_mode=False,
                                  opponent_type="balanced", matches_per_agent=2)
            with EvolutionTrainer(config) as trainer:
                trainer.train()
                assert trainer.best_genome is not None


class TestTrainingImprovesAbsolutely:
    """Guards against the population improving only relative to itself."""

    def test_trained_champion_beats_its_own_ancestor(self):
        """Self-referential fitness can rise while nothing gets stronger.

        Playing the final champion against the generation-0 champion is an
        absolute measure: it cannot be satisfied by the whole field drifting.
        """
        with tempfile.TemporaryDirectory() as tmp:
            config = TrainingConfig(
                population_size=12, elite_count=3, max_generations=8,
                num_workers=4, match_duration="short", runs_dir=tmp, seed=3,
                checkpoint_interval=100, curriculum_learning=False,
                diversity_preservation=False, tournament_matches=2,
                hall_of_fame_size=3,
            )
            with EvolutionTrainer(config) as trainer:
                trainer.population.initialize(seed=config.seed)
                start_genomes = trainer.population.get_population_weights()
                start_results = trainer._evaluate_population(start_genomes, 0)
                ancestor = np.array(
                    start_genomes[max(range(len(start_results)),
                                      key=lambda i: start_results[i].fitness)],
                    copy=True)

                trainer.train()
                champion = np.array(trainer.best_genome, copy=True)

        result = _run_head_to_head(0, champion, ancestor,
                                   matches_per_pair=20, seed=777)
        wins = result.metadata["agent1_wins"]
        losses = result.metadata["agent1_losses"]
        assert wins > losses, (
            f"trained champion went {wins}W/{losses}L against the "
            f"generation-0 champion: tournament fitness rose without "
            f"producing a genuinely stronger agent"
        )
