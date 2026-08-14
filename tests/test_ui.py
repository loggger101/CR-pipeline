"""Tests for the desktop UI.

Split in two: the operations layer and job runner are pure logic and always
tested; the widget tests need a display and skip without one, so the suite
still runs on a headless machine.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.models.policy import DEFAULT_POLICY_SPEC
from src.ui.arena_canvas import snapshot_from_engine
from src.ui.jobs import JobContext, JobRunner
from src.ui.operations import (
    MATCH_DURATIONS, SCRIPTED_OPPONENTS, TOURNAMENT_FORMATS,
    build_training_config, evaluate_head_to_head, list_runs,
    load_agent_genome, new_run_dir, play_match, resolve_runs_dir, run_training,
)


def _genome(seed: int = 0) -> np.ndarray:
    return DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(seed))


def _base_values(**overrides):
    values = dict(
        population_size=8, max_generations=2, elite_count=2, num_workers=2,
        seed=42, tournament_format="swiss", tournament_matches=2,
        hall_of_fame_size=2, match_duration="short",
        opponent_type="balanced", matches_per_agent=2, tournament_mode=True,
    )
    values.update(overrides)
    return values


def _run_to_completion(runner: JobRunner, target, timeout: float = 300):
    runner.start("test", target)
    deadline = time.time() + timeout
    events = []
    while runner.busy and time.time() < deadline:
        events.extend(runner.drain())
        time.sleep(0.02)
    runner.join(timeout=30)
    events.extend(runner.drain())
    return events


class TestJobRunner:
    """Background work must never block or silently swallow failures."""

    def test_runs_a_job_and_reports_the_result(self):
        runner = JobRunner()
        events = _run_to_completion(runner, lambda ctx: 21 * 2)
        done = [e for e in events if e.kind == "done"]
        assert len(done) == 1
        assert done[0].payload == 42

    def test_reports_failures_instead_of_hiding_them(self):
        """A crashed job must surface, not leave the UI waiting forever."""
        def explode(ctx):
            raise RuntimeError("boom")

        runner = JobRunner()
        events = _run_to_completion(runner, explode)
        errors = [e for e in events if e.kind == "error"]
        assert len(errors) == 1
        assert errors[0].payload["message"] == "boom"
        assert "RuntimeError" in errors[0].payload["traceback"]

    def test_refuses_a_second_concurrent_job(self):
        runner = JobRunner()
        runner.start("first", lambda ctx: time.sleep(0.4))
        assert runner.busy
        assert runner.start("second", lambda ctx: None) is False
        runner.join(timeout=10)

    def test_progress_and_log_events_reach_the_queue(self):
        def chatty(ctx: JobContext):
            ctx.log("working")
            ctx.progress({"step": 1})
            return "ok"

        runner = JobRunner()
        events = _run_to_completion(runner, chatty)
        kinds = [e.kind for e in events]
        assert "log" in kinds and "progress" in kinds and "done" in kinds

    def test_cancellation_is_visible_to_the_job(self):
        def waits(ctx: JobContext):
            for _ in range(400):
                if ctx.cancelled:
                    return "stopped"
                time.sleep(0.01)
            return "finished"

        runner = JobRunner()
        runner.start("waits", waits)
        time.sleep(0.1)
        runner.cancel()
        runner.join(timeout=15)
        done = [e for e in runner.drain() if e.kind == "done"]
        assert done and done[0].payload == "stopped"

    def test_drain_is_bounded(self):
        """A burst of events must not let one drain stall the UI thread."""
        from src.ui.jobs import JobEvent

        runner = JobRunner()
        for index in range(50):
            runner.post(JobEvent("log", index))

        assert len(runner.drain(limit=10)) == 10
        assert len(runner.drain(limit=100)) == 40   # the rest, then empty
        assert runner.drain() == []


class TestConfigBuilding:
    """UI field values must be validated before reaching the trainer."""

    def test_builds_a_usable_config(self):
        config = build_training_config(_base_values(), "runs/x")
        assert config.population_size == 8
        assert config.tournament_mode is True
        assert config.tournament_format == "swiss"
        assert config.runs_dir == "runs/x"

    @pytest.mark.parametrize("overrides,message", [
        ({"population_size": "many"}, "whole number"),
        ({"population_size": 1}, "at least 2"),
        ({"elite_count": 8}, "smaller than population"),
        ({"tournament_format": "knockout"}, "unknown tournament format"),
        ({"opponent_type": "nobody"}, "unknown opponent"),
        ({"match_duration": "eternal"}, "unknown match duration"),
        ({"max_generations": 0}, "at least 1"),
    ])
    def test_rejects_bad_input_with_a_readable_message(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            build_training_config(_base_values(**overrides), "runs/x")

    def test_choices_match_what_the_pipeline_accepts(self):
        assert "swiss" in TOURNAMENT_FORMATS
        assert "balanced" in SCRIPTED_OPPONENTS
        assert "short" in MATCH_DURATIONS


class TestRunDirectories:

    def test_each_run_gets_its_own_directory(self):
        """The trainer writes straight into runs_dir, so runs must not share
        one or they overwrite each other."""
        with tempfile.TemporaryDirectory() as tmp:
            first = new_run_dir(tmp)
            second = new_run_dir(tmp)
            assert first != second
            assert os.path.isdir(first) and os.path.isdir(second)

    def test_listing_ignores_checkpoint_folders(self):
        """gen_0001/ lives inside a run; it is not a run."""
        with tempfile.TemporaryDirectory() as tmp:
            run = new_run_dir(tmp)
            os.makedirs(os.path.join(run, "gen_0001"), exist_ok=True)
            os.makedirs(os.path.join(run, "best"), exist_ok=True)
            runs = list_runs(tmp)
            assert [r["run_id"] for r in runs] == [os.path.basename(run)]

    def test_listing_a_missing_directory_is_empty(self):
        assert list_runs("no/such/place") == []

    def test_runs_dir_defaults_beside_the_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_runs_dir(tmp)
            assert resolved == os.path.join(tmp, "runs")
            assert os.path.isdir(resolved)

    def test_explicit_runs_dir_is_used_verbatim(self):
        """--runs-dir must point at that folder, not at <parent>/runs."""
        with tempfile.TemporaryDirectory() as tmp:
            elsewhere = os.path.join(tmp, "somewhere", "else")
            resolved = resolve_runs_dir(tmp, elsewhere)
            assert resolved == elsewhere
            assert os.path.isdir(elsewhere)


class TestAgentLoading:

    def test_rejects_a_torch_network_checkpoint(self):
        """Those cannot be played by the simulator; say so plainly."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "net.pt")
            torch.save({"weights": np.zeros(9_277_223, dtype=np.float32)}, path)
            with pytest.raises(ValueError, match="Torch network"):
                load_agent_genome(path)

    def test_loads_a_genome_checkpoint(self):
        import torch
        genome = _genome(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            torch.save({"genome": genome, "param_kind": "genome"}, path)
            assert np.array_equal(load_agent_genome(path), genome)


class TestMatchPlayback:

    def test_produces_replayable_frames(self):
        runner = JobRunner()
        events = _run_to_completion(
            runner,
            lambda ctx: play_match(ctx, _genome(1), opponent="balanced",
                                   seed=5, match_duration_ticks=400),
        )
        done = [e for e in events if e.kind == "done"]
        assert done, [e.payload for e in events if e.kind == "error"]
        recording = done[0].payload
        assert len(recording) > 2
        assert recording.winner in {"player", "opponent", "tie", "none"}

    def test_frames_carry_everything_the_arena_draws(self):
        runner = JobRunner()
        events = _run_to_completion(
            runner,
            lambda ctx: play_match(ctx, _genome(1), opponent="balanced",
                                   seed=5, match_duration_ticks=400),
        )
        recording = [e for e in events if e.kind == "done"][0].payload
        frame = recording.frames[len(recording.frames) // 2]
        for key in ("tick", "towers", "units", "player_elixir",
                    "opponent_elixir", "player_crowns", "overtime"):
            assert key in frame
        assert len(frame["towers"]) == 6
        for tower in frame["towers"]:
            assert {"col", "row", "owner", "hp", "max_hp", "alive", "king"} <= set(tower)

    def test_snapshot_excludes_dead_and_building_units(self):
        from src.env.sim.engine import SimulationEngine

        engine = SimulationEngine(seed=2, record_replay=False)
        engine.reset()
        engine._spawn_unit("knight", 3.0, 4.0, "player")
        engine.player_units[-1].take_damage(1e9)

        snapshot = snapshot_from_engine(engine)
        assert snapshot["units"] == []          # the corpse is not drawn
        assert len(snapshot["towers"]) == 6     # towers still are


class TestTrainingJob:
    """The training job is what the Train tab drives."""

    def test_streams_progress_and_saves_an_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = build_training_config(
                _base_values(max_generations=2), new_run_dir(tmp))
            runner = JobRunner()
            events = _run_to_completion(
                runner, lambda ctx: run_training(ctx, config))

            errors = [e for e in events if e.kind == "error"]
            assert not errors, errors[0].payload["traceback"] if errors else ""

            progress = [e.payload for e in events if e.kind == "progress"]
            assert len(progress) == 2
            for snapshot in progress:
                for key in ("generation", "best_fitness", "mean_fitness",
                            "champion_elo", "champion_record"):
                    assert key in snapshot

            result = [e for e in events if e.kind == "done"][0].payload
            assert result["generations_completed"] == 2
            assert result["best_agent_path"] is not None
            assert os.path.exists(result["best_agent_path"])
            # Tournament mode ranks across generations by ELO, so the label
            # must not claim the number is a fitness score.
            assert result["best_score_label"] == "ELO"

    def test_writes_a_run_level_history(self):
        """Tools read the run directory, not just checkpoint folders."""
        with tempfile.TemporaryDirectory() as tmp:
            config = build_training_config(
                _base_values(max_generations=2), new_run_dir(tmp))
            runner = JobRunner()
            _run_to_completion(runner, lambda ctx: run_training(ctx, config))

            listed = list_runs(tmp)
            assert len(listed) == 1
            assert listed[0]["generations"] == 2
            assert listed[0]["has_agent"] is True

    def test_stopping_ends_the_run_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = build_training_config(
                _base_values(max_generations=40), new_run_dir(tmp))
            runner = JobRunner()
            runner.start("training", lambda ctx: run_training(ctx, config))

            deadline = time.time() + 120
            while runner.busy and time.time() < deadline:
                if any(e.kind == "progress" for e in runner.drain()):
                    runner.cancel()
                    break
                time.sleep(0.02)
            runner.join(timeout=120)

            done = [e for e in runner.drain() if e.kind == "done"]
            assert done, "training did not finish after cancel"
            result = done[0].payload
            assert result["cancelled"] is True
            assert result["generations_completed"] < 40


class TestEvaluationJob:

    def test_head_to_head_reports_a_full_record(self):
        runner = JobRunner()
        events = _run_to_completion(
            runner,
            lambda ctx: evaluate_head_to_head(ctx, _genome(1), _genome(2),
                                              matches=4),
        )
        result = [e for e in events if e.kind == "done"][0].payload
        assert result["matches"] > 0
        assert (result["a_wins"] + result["a_draws"]
                + result["a_losses"]) == result["matches"]
        assert 0.0 <= result["a_win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Widget tests (need a display)
# ---------------------------------------------------------------------------

def _display_available() -> bool:
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:
        return False


needs_display = pytest.mark.skipif(
    not _display_available(), reason="no display available for Tk")


@needs_display
class TestWindow:

    @pytest.fixture
    def app(self):
        from src.ui.app import CRPipelineApp
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "runs"), exist_ok=True)
            window = CRPipelineApp(project_root=tmp)
            window.update_idletasks()
            yield window
            window.destroy()

    def test_opens_with_the_expected_tabs(self, app):
        tabs = [app.notebook.tab(i, "text") for i in app.notebook.tabs()]
        assert tabs == ["Train", "Watch", "Runs", "Agents"]

    def test_runs_dir_defaults_beside_the_project(self, app):
        assert app.runs_dir == os.path.join(str(app.project_root), "runs")
        assert os.path.isdir(app.runs_dir)

    def test_every_tab_renders(self, app):
        for index in range(len(app.notebook.tabs())):
            app.notebook.select(index)
            app.update_idletasks()
            app.update()

    def test_progress_updates_the_headline_and_chart(self, app):
        app.training_tab._on_generation({
            "generation": 4, "total_generations": 10, "best_fitness": 1.5,
            "mean_fitness": 0.9, "champion_elo": 1575.0,
            "champion_record": "6W/0D/2L",
        })
        app.update_idletasks()
        headline = app.training_tab.headline.get()
        assert "gen 4/10" in headline
        assert "1575" in headline
        assert app.training_tab.history["best"] == [1.5]

    def test_arena_draws_a_snapshot(self, app):
        app.watch_tab.arena.show({
            "tick": 10, "player_elixir": 5.0, "opponent_elixir": 5.0,
            "player_crowns": 0, "opponent_crowns": 0, "overtime": False,
            "towers": [{"col": 2, "row": 4, "owner": "player", "hp": 10,
                        "max_hp": 10, "alive": True, "king": False}],
            "units": [{"col": 3, "row": 3, "owner": "player", "type": "knight",
                       "hp": 5, "max_hp": 10, "air": False}],
            "winner": None, "reason": None,
        })
        app.update_idletasks()
        assert len(app.watch_tab.arena.find_all()) > 5

    def test_bad_settings_do_not_start_a_job(self, app, monkeypatch):
        shown = {}
        monkeypatch.setattr("src.ui.app.messagebox.showerror",
                            lambda title, msg: shown.update(title=title, msg=msg))
        app.training_tab.fields["population_size"].var.set("lots")
        app.training_tab._start()
        assert "population size" in shown.get("msg", "")
        assert not app.jobs.busy
