"""Tests for the pygame arena window (src/viz/pygame_viewer.py).

The viewer's contract is that *update* and *render* are independent: ``step``
mutates only engine state, ``render_frame`` reads it. That split makes the
whole thing testable headless -- no monitor needed -- which is exactly how CI
exercises it (and how ``crp watch --frames N`` smoke-tests a full match).

These tests follow the pygame suite's own discipline: skip when the display/SDL
device is absent rather than failing, and drive bounded frame counts so nothing
can hang.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

# Skip (don't fail) when pygame or a display device is unavailable -- the same
# pattern the upstream pygame test suite uses for optional hardware.
pygame = pytest.importorskip("pygame")


def _headless():
    """Force the dummy SDL driver *before* any viewer import in this process."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    return True


@pytest.fixture()
def window():
    """A headless arena window with two heuristic opponents, torn down after use."""
    _headless()
    from src.viz.pygame_viewer import ArenaWindow

    win = ArenaWindow(headless=True)   # random vs balanced by default
    yield win


class TestViewerLifecycle:
    def test_step_advances_the_engine(self, window):
        start_tick = window.engine.tick
        assert window.step() in (True, False)  # a short match may end immediately
        if not window.engine.terminated and not window.engine.truncated:
            assert window.engine.tick > start_tick

    def test_step_returns_false_when_match_over(self, window):
        engine = window.engine
        while not (engine.terminated or engine.truncated) and engine.tick < 3000:
            if not window.step():
                break
        # Once the match is over, step must refuse to advance it further.
        assert window.step() is False

    def test_new_match_resets_state(self, window):
        while not (window.engine.terminated or window.engine.truncated) \
                and window.engine.tick < 50:
            if not window.step():
                break
        window.new_match()
        assert window.engine.tick == 0
        assert not window.engine.terminated

    def test_speed_change_is_applied(self, window):
        window.speed = 25.0
        assert window.speed == 25.0


class TestViewerRendering:
    """render_frame must produce a sane frame for any live engine state."""

    def test_frame_has_expected_size_and_varied_pixels(self, window):
        from src.viz.pygame_viewer import WIN_W, WIN_H

        frame = window.render_frame()
        assert frame.get_size() == (WIN_W, WIN_H)
        # The arena has a river band, bridges, six towers and HUD bars -- the
        # frame must not be one flat colour. Count distinct colours cheaply by
        # sampling: at least 4 different RGB values across the surface.
        import pygame as pg
        arr = np.asarray(pg.surfarray.array3d(frame))  # shape (W, H, 3)
        h = arr.shape[1]
        sample_rows = [arr[:, r] for r in (min(10, h - 1), min(h // 2, h - 1), min(int(h * 0.85), h - 1))]
        unique = {tuple(int(c) for c in px) for row in sample_rows for px in row[::97]}
        assert len(unique) >= 4

    def test_rendering_survives_a_mid_match_state(self, window):
        """Drive the engine a while (units on board), then render -- no crash."""
        for _ in range(60):
            if not window.step():
                break
        frame = window.render_frame()
        assert frame.get_size()[0] > 100

    def test_rendering_survives_terminated_match(self, window):
        engine = window.engine
        while not (engine.terminated or engine.truncated) and engine.tick < 3000:
            if not window.step():
                break
        frame = window.render_frame()   # the "match over" overlay path
        assert frame is not None


class TestViewerSources:
    """Both sides must be drivable by genomes *and* profiles."""

    def test_genome_side_plays(self):
        _headless()
        from src.viz.pygame_viewer import ArenaWindow
        from src.models.policy import DEFAULT_POLICY_SPEC, compile_genome

        genome = DEFAULT_POLICY_SPEC.random_genome(np.random.RandomState(7))
        win = ArenaWindow(player_source=("genome", compile_genome(genome)),
                          opponent_source=("profile", "balanced"), headless=True)
        try:
            ticks = 0
            for _ in range(120):
                if not win.step():
                    break
                ticks += 1
            assert ticks > 0 or (win.engine.terminated or win.engine.truncated)
        finally:
            pygame.quit()

    def test_unknown_profile_rejected(self):
        _headless()
        from src.viz.pygame_viewer import ArenaWindow

        with pytest.raises(ValueError, match="unknown opponent profile"):
            # Construction succeeds (lazy resolution); the failure surfaces on
            # the first action -- exercise it explicitly.
            win = ArenaWindow(player_source=("profile", "not_a_profile"),
                              headless=True)
            try:
                win._side_action(("profile", "not_a_profile"), "player")
            finally:
                pygame.quit()

    def test_from_models_without_checkpoints_uses_profiles(self):
        _headless()
        from src.viz.pygame_viewer import ArenaWindow

        win = ArenaWindow.from_models(None, None, profile_a="greedy",
                                      profile_b="aggressive")
        try:
            assert win.player_source == ("profile", "greedy")
            assert win.opponent_source == ("profile", "aggressive")
        finally:
            pygame.quit()


class TestRunArenaHeadless:
    """The CLI entry path must terminate on its own in headless mode."""

    def test_run_arena_bounded_frames(self):
        _headless()
        from src.viz.pygame_viewer import run_arena

        rc = run_arena(profile_a="random", profile_b="random", frames=30)
        assert rc == 0
        pygame.quit()
