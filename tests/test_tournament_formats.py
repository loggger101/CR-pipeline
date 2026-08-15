"""Every tournament format must finish, and must run its matches on the pool.

These properties had no coverage: the existing tests only asserted that the
``TournamentFormat`` members exist, so two formats that never terminated and
three that ran a whole generation on the calling thread all passed a green
suite. Both faults reached the desktop app as a frozen window.

The matches themselves are stubbed. What is under test is the scheduling --
termination, bookkeeping, and whether work is handed to the worker pool -- not
the simulation, which has its own tests.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.env.sim.parallel_runner import MatchResult, ParallelRunner
from src.models.policy import DEFAULT_POLICY_SPEC
from src.train import FitnessEvaluator, TournamentFormat, TournamentRunner

ALL_FORMATS = [
    TournamentFormat.SWISS,
    TournamentFormat.ROUND_ROBIN,
    TournamentFormat.SINGLE_ELIMINATION,
    TournamentFormat.DOUBLE_ELIMINATION,
    TournamentFormat.LEAGUE,
]


class StubRunner:
    """Stands in for ParallelRunner, recording how matchups were dispatched.

    ``run_head_to_head`` executes a matchup inline on the caller's thread;
    ``run_pairings`` is the batched call that puts it on the worker pool. The
    counts here are what distinguish the two.
    """

    #: Guards against a format that never terminates hanging the test run.
    MAX_MATCHUPS = 5000

    def __init__(self, agent1_wins=2, agent2_wins=0):
        self.matchups = 0
        self.inline_calls = 0
        self.batched_calls = 0
        self.batch_sizes = []
        self._agent1_wins = agent1_wins
        self._agent2_wins = agent2_wins

    def _result(self):
        self.matchups += 1
        if self.matchups > self.MAX_MATCHUPS:
            raise RuntimeError(
                f"tournament issued more than {self.MAX_MATCHUPS} matchups; "
                f"it is not converging"
            )
        return MatchResult(
            agent_id="stub", fitness=1.0,
            wins=self._agent1_wins, losses=self._agent2_wins,
            metadata={
                "agent1_wins": self._agent1_wins, "agent1_draws": 0,
                "agent1_losses": self._agent2_wins, "agent1_towers": 2,
                "agent1_duration": 100,
                "agent2_wins": self._agent2_wins, "agent2_draws": 0,
                "agent2_losses": self._agent1_wins, "agent2_towers": 0,
                "agent2_duration": 100,
            },
        )

    def run_head_to_head(self, agent1_weights, agent2_weights,
                         matches_per_pair=4, seed=42, deck=None,
                         opponent_deck=None):
        self.inline_calls += 1
        return self._result()

    def run_pairings(self, pairings, weights_list, matches_per_pair=4, seed=42,
                     deck=None, opponent_deck=None):
        self.batched_calls += 1
        self.batch_sizes.append(len(pairings))
        return [self._result() for _ in pairings]


def _run_with_timeout(call, timeout=30.0):
    """Run ``call``, failing rather than hanging if it does not return."""
    box = {}

    def target():
        try:
            box["value"] = call()
        except BaseException as exc:      # re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"tournament did not finish within {timeout:.0f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.fixture
def entrants():
    ids = [f"agent_{i}" for i in range(8)]
    weights = [np.zeros(DEFAULT_POLICY_SPEC.num_params) for _ in ids]
    return ids, weights


class TestFormatsTerminate:
    """No format may run forever."""

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_format_finishes(self, fmt, entrants):
        ids, weights = entrants
        runner = TournamentRunner(StubRunner())
        result = _run_with_timeout(
            lambda: runner.run_tournament(ids, weights, format=fmt,
                                          matches_per_pair=2))
        assert result is not None

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_every_entrant_is_ranked(self, fmt, entrants):
        ids, weights = entrants
        runner = TournamentRunner(StubRunner())
        result = _run_with_timeout(
            lambda: runner.run_tournament(ids, weights, format=fmt,
                                          matches_per_pair=2))
        assert {aid for aid, _ in result.rankings} == set(ids)
        assert set(result.agent_stats) == set(ids)

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_odd_field_finishes(self, fmt):
        """A field that cannot be halved evenly must still resolve."""
        ids = [f"agent_{i}" for i in range(7)]
        weights = [np.zeros(DEFAULT_POLICY_SPEC.num_params) for _ in ids]
        runner = TournamentRunner(StubRunner())
        result = _run_with_timeout(
            lambda: runner.run_tournament(ids, weights, format=fmt,
                                          matches_per_pair=2))
        assert len(result.rankings) == 7

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_elimination_bracket_shrinks(self, fmt, entrants):
        """Match count stays proportional to the field, not unbounded.

        Single and double elimination previously re-paired every survivor
        against a bye, so the field never shrank and both the match count and
        the bracket grew without limit.
        """
        ids, weights = entrants
        stub = StubRunner()
        runner = TournamentRunner(stub)
        _run_with_timeout(lambda: runner.run_tournament(
            ids, weights, format=fmt, matches_per_pair=2))
        # Round-robin's n*(n-1)/2 = 28 is the densest legitimate schedule.
        assert stub.matchups <= 28


class TestFormatsUseThePool:
    """Matches belong on worker processes, not the calling thread."""

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_no_matchup_runs_inline(self, fmt, entrants):
        ids, weights = entrants
        stub = StubRunner()
        runner = TournamentRunner(stub)
        _run_with_timeout(lambda: runner.run_tournament(
            ids, weights, format=fmt, matches_per_pair=2))

        assert stub.matchups > 0, "the tournament played no matches at all"
        assert stub.inline_calls == 0, (
            f"{fmt.name} ran {stub.inline_calls} matchups on the calling "
            f"thread; they must be batched onto the pool"
        )
        assert stub.batched_calls > 0

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_batches_are_whole_rounds(self, fmt, entrants):
        """A round goes out in one batch, so the pool is filled in one go."""
        ids, weights = entrants
        stub = StubRunner()
        runner = TournamentRunner(stub)
        _run_with_timeout(lambda: runner.run_tournament(
            ids, weights, format=fmt, matches_per_pair=2))
        assert max(stub.batch_sizes) > 1


class TestCarriedRatings:
    """ELO is the trainer's cross-generation measure, so it must carry over."""

    @pytest.mark.parametrize("fmt", ALL_FORMATS, ids=lambda f: f.name)
    def test_initial_elo_is_honoured(self, fmt, entrants):
        ids, weights = entrants
        # A rating far from the default survives only if it was read at all;
        # every non-Swiss format used to reset the whole field to 1500.
        carried = {ids[0]: 2400.0}
        runner = TournamentRunner(StubRunner())
        result = _run_with_timeout(lambda: runner.run_tournament(
            ids, weights, format=fmt, matches_per_pair=2,
            initial_elo=carried))
        assert result.elo_ratings[ids[0]] > 2000.0


class TestPoolLifecycle:
    """One runner, one pool."""

    def test_start_is_idempotent(self):
        runner = ParallelRunner(num_workers=2)
        try:
            runner.start()
            first = runner.pool
            runner.start()
            assert runner.pool is first, (
                "start() replaced a running pool, orphaning its workers"
            )
        finally:
            runner.shutdown()

    def test_shared_runner_is_not_shut_down_by_the_evaluator(self):
        """A pool passed in belongs to the caller."""
        runner = ParallelRunner(num_workers=2)
        runner.start()
        try:
            evaluator = FitnessEvaluator(runner=runner)
            assert evaluator.runner is runner
            evaluator.shutdown()
            assert runner.pool is not None
        finally:
            runner.shutdown()

    def test_evaluator_closes_a_pool_it_created(self):
        evaluator = FitnessEvaluator(num_workers=2)
        assert evaluator.runner.pool is not None
        evaluator.shutdown()
        assert evaluator.runner.pool is None
