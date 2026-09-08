"""Pygame arena window: watch a simulation match play out live.

The training pipeline runs thousands of matches per generation with no way to
*see* them -- fitness numbers are all you get, and when evolution stalls it is
impossible to tell whether the agents are playing badly or the sim is broken.
This module closes that gap: it drives a ``SimulationEngine`` tick by tick in
a pygame window (or headlessly) while both sides act through real policies --
evolved genomes loaded from checkpoints, heuristic opponent profiles, or pure
random play.

Design follows the standard pygame discipline:

* **Update is separated from render.** :meth:`ArenaWindow.step` mutates only
  engine state; :meth:`render_frame` reads it and blits. That split is what
  makes headless testing possible -- drive ``step`` thousands of times with no
  window at all, then call ``render_frame`` on a dummy display to verify the
  drawing code produces sane pixels.
* **Headless first.** With ``SDL_VIDEODRIVER=dummy`` (set *before* importing
  pygame) everything works without a monitor: CI can smoke-test the whole
  viewer, and ``--headless --frames N`` runs a bounded match in a terminal.
* **Event pump every frame** so QUIT/keys always arrive; ``clock.tick`` caps
  the framerate (never ``time.sleep``).

Usage from the CLI::

    python scripts/crp.py watch                          # random vs random
    python scripts/crp.py watch --model-a runs/<run>/best/best_agent.pt \
                                --profile-b balanced      # evolved vs heuristic
    python scripts/crp.py watch --headless --frames 300   # no window, bounded
    python scripts/crp.py watch --seed 42                 # R replays exactly this game

Keys: space = pause/resume, R = restart match (same seed), +/- = speed, Q/ESC = quit.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _require_pygame():
    """Import pygame lazily so the rest of ``src.viz`` works without it.

    The dummy driver must be in place *before* the first import when running
    headless; callers set ``CRP_HEADLESS=1`` (or pass ``headless=True``) and
    this installs it here, which is still before any pygame module loads.
    """
    if os.environ.get("CRP_PYGAME_NO_IMPORT") == "1":  # test hook: force failure path
        raise ImportError("pygame import disabled by CRP_PYGAME_NO_IMPORT=1")
    try:
        import pygame as pg
    except ImportError as exc:
        raise ImportError(
            "pygame is required for the arena window. Install it with "
            "`pip install pygame` (it is listed in requirements.txt)."
        ) from exc
    return pg


# ── Layout constants ────────────────────────────────────────────────────────
GRID_COLS = 8
GRID_ROWS = 6
CELL_W, CELL_H = 90, 120          # arena pixels per grid cell (tall: CR is vertical)
HUD_TOP, HUD_BOTTOM = 34, 56      # reserved for elixir/hand and status bars
ARENA_X, ARENA_Y = 8, HUD_TOP + 8

WIN_W = ARENA_X * 2 + GRID_COLS * CELL_W          # 744
WIN_H = ARENA_Y + GRID_ROWS * CELL_H + HUD_BOTTOM  # 906

# Palette (RGB) -- blue is the bottom player, red the top opponent.
C_BG        = (18, 22, 30)
C_RIVER     = (45, 70, 110)
C_BRIDGE    = (120, 96, 60)
C_GRID      = (34, 40, 52)
C_PLAYER    = (80, 150, 255)
C_OPPONENT  = (255, 90, 90)
C_KING_RING = (255, 215, 60)
C_TEXT      = (230, 235, 245)
C_DIM       = (120, 128, 145)
C_READY     = (90, 220, 130)
C_COOLDOWN  = (70, 76, 90)

# Opponent profile name -> action function in the parallel runner. The lookup
# is explicit rather than getattr so a typo fails at construction time with a
# readable message instead of an AttributeError deep inside a match loop.
_PROFILE_ACTIONS: dict = {}


def _profile_action_fn(name: str):
    """Resolve an opponent profile name to its action function (imported once)."""
    global _PROFILE_ACTIONS
    if not _PROFILE_ACTIONS:
        from ..env.sim import parallel_runner as pr
        _PROFILE_ACTIONS.update({
            "random":     lambda eng, prof=None: pr._random_opponent_action(eng),
            "greedy":     lambda eng, prof=None: pr._greedy_opponent_action(eng),
            "balanced":   lambda eng, prof=pr.OPPONENT_PROFILES["balanced"]:
                           pr._heuristic_opponent_action(eng, prof),
            "aggressive": lambda eng, prof=pr.OPPONENT_PROFILES["aggressive"]:
                           pr._heuristic_opponent_action(eng, prof),
            "defensive":  lambda eng, prof=pr.OPPONENT_PROFILES["defensive"]:
                           pr._heuristic_opponent_action(eng, prof),
        })
    if name not in _PROFILE_ACTIONS:
        raise ValueError(
            f"unknown opponent profile {name!r}; expected one of "
            f"{sorted(_PROFILE_ACTIONS)}")
    return _PROFILE_ACTIONS[name]


class ArenaWindow:
    """Drives a simulation match and renders it to a pygame window.

    Each side is controlled by an *action source*: either a compiled policy
    genome (the real evolved net, run through the same ``_policy_action`` path
    training uses) or a named heuristic profile / random play. Both sides can
    be any combination of the two.

    Args:
        engine_factory: Zero-arg callable returning a *fresh* engine per match
            (so restarts re-seed cleanly). Defaults to a standard 180-second
            regulation + sudden-death overtime match with replay off (the
            window is itself the visualization; storing frames would double
            memory for nothing here).
        player_source / opponent_source: ``("genome", compiled)`` or
            ``("profile", name)``. See :meth:`from_models` for loading genomes
            from checkpoint files.
        speed: Ticks per second (engine default is 10; the window renders at up
            to 60 fps and steps whole ticks, so fractional speeds are fine).
        headless: Run with ``SDL_VIDEODRIVER=dummy`` -- no visible window.
    """

    def __init__(self, engine_factory=None, player_source=("profile", "random"),
                 opponent_source=("profile", "random"), speed: float = 10.0,
                 headless: bool = False, match_seed: Optional[int] = None):
        if headless or os.environ.get("CRP_HEADLESS") == "1":
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.pg = _require_pygame()

        # Audio degrades gracefully on machines without a sound device (and in
        # CI): pre_init before init, then check -- the canonical two-line guard.
        if not headless:
            try:
                self.pg.mixer.pre_init(44100, -16, 2, 512)
            except Exception:
                pass
        self.pg.display.init()

        bestdepth = self.pg.display.mode_ok((WIN_W, WIN_H), 0, 32)
        self.screen = self.pg.display.set_mode((WIN_W, WIN_H), 0, bestdepth)
        # Fonts only after set_mode (they need the display format to exist).
        try:
            font_big = self.pg.font.SysFont(None, 26)
            font_small = self.pg.font.SysFont(None, 18)
        except Exception:
            font_big = font_small = None
        if font_small is None:  # no fonts at all (minimal SDL build): blank text
            def _blank(text, color=C_TEXT):
                return self.pg.Surface((2, 2))
            font_big = font_small = type("F", (), {"render": lambda f, t, c, col: _blank(t)})()

        self.font_big = font_big
        self.font_small = font_small

        from ..env.sim.engine import SimulationEngine
        if engine_factory is None:
            def engine_factory():  # noqa: E306 - default factory closure
                return SimulationEngine(record_replay=False)
        self._engine_factory = engine_factory
        self.player_source = player_source
        self.opponent_source = opponent_source
        self.speed = float(speed)

        # Fixed seed for this window's match (R replays it). When None, the
        # first engine construction decides: seeded factories are remembered;
        # unseeded ones stay fresh on restart.
        self._match_seed = int(match_seed) if match_seed is not None else None

        self.paused = False
        self.engine = None
        self.new_match()

    # ── Construction helpers ────────────────────────────────────────────────

    @classmethod
    def from_models(cls, model_a: Optional[str], model_b: Optional[str] = None,
                    profile_a: str = "random", profile_b: str = "balanced",
                    speed: float = 10.0, headless: bool = False,
                    match_seed: Optional[int] = None) -> "ArenaWindow":
        """Build a window from checkpoint files and/or opponent profiles.

        ``model_a``/``model_b`` are paths to agent checkpoints (or ``None``, in
        which case the named profile is used instead). Genomes load through the
        same validated path training uses, so a corrupted or wrong-shape file
        fails here with a clear error rather than producing silent random play.

        ``match_seed`` pins this window's match: R replays the exact same game
        (same deck shuffles and opponent draws), which is what makes it useful
        for debugging one specific matchup. Omit it for fresh games each time.
        """
        from ..serialization import load_agent_genome
        from ..models.policy import compile_genome

        def source(model_path: Optional[str], profile_name: str):
            if model_path is not None:
                genome = load_agent_genome(str(model_path))
                return ("genome", compile_genome(genome))
            return ("profile", profile_name)

        return cls(player_source=source(model_a, profile_a),
                   opponent_source=source(model_b, profile_b),
                   speed=speed, headless=headless, match_seed=match_seed)

    # ── Match lifecycle ─────────────────────────────────────────────────────

    def new_match(self):
        """Start (or restart) a match with a fresh engine.

        Restarts re-seed the original seed, so ``R`` replays *the same* game --
        useful for debugging one specific match instead of getting a new random
        one every time. A different match comes from quitting and relaunching.
        """
        self.engine = self._engine_factory()
        # Fresh per match, so an ended game never flashes over its replacement.
        self._last_result_info: Optional[dict] = None
        if self._match_seed is not None:
            # reset(seed=...) re-derives the engine's RNG (and deck shuffles)
            # from that seed, so every restart reproduces this window's match.
            self.engine.reset(seed=self._match_seed)
        elif getattr(self.engine, "_initial_seed", None) is not None:
            # First construction with a seeded factory: remember it for R.
            self._match_seed = int(self.engine._initial_seed)
        self.paused = False

    def _side_action(self, source: Tuple[str], side: str):
        """One action for ``side`` from its configured source."""
        kind, payload = source
        if kind == "genome":
            from ..env.sim.parallel_runner import _policy_action
            return _policy_action(payload, self.engine, side)
        # profile (or random/greedy shorthands): heuristic play.
        fn = _profile_action_fn(payload)
        return fn(self.engine)

    def step(self) -> bool:
        """Advance the engine one tick; returns False once the match is over."""
        if self.engine.terminated or self.engine.truncated:
            return False
        player_act = self._side_action(self.player_source, "player")
        opponent_act = self._side_action(self.opponent_source, "opponent")
        result = self.engine.step(player_act, opponent_act)
        # Remember the termination info (winner/reason) so the overlay can show
        # *who* won instead of just that it ended. The engine returns this on
        # the step-result; nothing else stores it after the fact.
        try:
            if result is not None and getattr(result, "info", None):
                self._last_result_info = dict(result.info)
        except Exception:
            pass
        return not (self.engine.terminated or self.engine.truncated)

    # ── Rendering (pure reads of engine state; safe headless) ───────────────

    def _cell_px(self, col: float, row: float) -> Tuple[int, int]:
        """Grid coordinates to pixel centre."""
        return (ARENA_X + int((col + 0.5) * CELL_W),
                ARENA_Y + int((row + 0.5) * CELL_H))

    def render_frame(self):
        """Draw the current engine state; returns a copy of the frame surface.

        Returning the surface (rather than only blitting it) is what lets tests
        assert on pixels without polling the display, and makes headless runs
        verifiable: ``crp watch --headless --frames N`` exercises this path end
        to end in CI with no monitor attached.
        """
        pg = self.pg
        eng = self.engine
        screen = self.screen
        screen.fill(C_BG)

        # River band between rows 2 and 3 (the two bridge lanes cross it).
        river_top = ARENA_Y + int(2 * CELL_H)
        river_h = int(CELL_H)   # row-3 cell height spans the water visually
        pg.draw.rect(screen, C_RIVER,
                     (ARENA_X, river_top, GRID_COLS * CELL_W, river_h))
        for bc in (3, 4):  # bridges: wood planks across the river at cols 3-4
            bx = ARENA_X + int(bc * CELL_W)
            pg.draw.rect(screen, C_BRIDGE, (bx + 10, river_top, CELL_W - 20, river_h))

        # Grid lines.
        for c in range(GRID_COLS + 1):
            x = ARENA_X + c * CELL_W
            pg.draw.line(screen, C_GRID, (x, ARENA_Y), (x, ARENA_Y + GRID_ROWS * CELL_H))
        for r in range(GRID_ROWS + 1):
            y = ARENA_Y + r * CELL_H
            pg.draw.line(screen, C_GRID, (ARENA_X, y), (ARENA_X + GRID_COLS * CELL_W, y))

        # Towers first so units draw on top of them.
        for towers, color in ((eng.opponent_towers, C_OPPONENT),
                              (eng.player_towers, C_PLAYER)):
            for t in towers:
                if not t.is_alive:
                    continue
                cx, cy = self._cell_px(t.col, t.row)
                rad = int(min(CELL_W, CELL_H) * 0.38)
                pg.draw.circle(screen, color, (cx, cy), rad)
                if t.is_king:
                    ring_color = C_KING_RING if t.is_active else C_DIM
                    pg.draw.circle(screen, ring_color, (cx, cy), rad + 4, width=3)
                self._hp_bar(cx, cy - rad - 8, rad * 2, t.hp / t.max_hp if t.max_hp else 0.0)

        # Units: filled circles; air units get a white halo so flyers read at a glance.
        for units, color in ((eng.opponent_units, C_OPPONENT),
                             (eng.player_units, C_PLAYER)):
            for u in units:
                if not u.is_alive or u.is_building:
                    continue
                cx, cy = self._cell_px(u.col, u.row)
                rad = int(min(CELL_W, CELL_H) * 0.22)
                if u.is_air:
                    pg.draw.circle(screen, (235, 238, 245), (cx, cy), rad + 3)
                pg.draw.circle(screen, color, (cx, cy), rad)
                self._hp_bar(cx, cy - rad - 6, int(rad * 1.8), u.hp / u.max_hp if u.max_hp else 0.0)

        # HUD: elixir bars + hand at the bottom (player), mirrored top (opponent).
        self._draw_hud(eng.player_elixir, eng.opponent_elixir, eng.elixir_max,
                       eng.player_hand, eng.player_cooldowns, side="bottom")
        self._draw_hud(eng.opponent_elixir, eng.player_elixir, eng.elixir_max,
                       eng.opponent_hand, eng.opponent_cooldowns, side="top")

        # Status line: clock, crowns, overtime (+double elixir while it lasts).
        total = eng.match_duration_ticks + (eng.overtime_ticks if eng.is_overtime else 0)
        secs_left = max(0, (total - eng.tick)) / float(eng.TICKS_PER_SECOND)
        ot = ""
        if eng.is_overtime:
            ot = "  OVERTIME"
            if getattr(eng, "double_elixir_overtime", True):
                ot += " (2x elixir)"
        status = f"{int(secs_left // 60)}:{secs_left % 60:04.1f}   crowns {eng.player_trophies}-{eng.opponent_trophies}{ot}"
        if self.font_big is not None and hasattr(self.font_big, "render"):
            surf = self.font_small.render(status, True, C_TEXT)
            screen.blit(surf, (ARENA_X + 2, WIN_H - HUD_BOTTOM + 18))
        if eng.terminated or eng.truncated:
            # Winner comes from the step-result info captured in step() -- the
            # engine does not store it anywhere else after termination.
            reason_map = {
                "king_tower_destroyed": "by king tower destruction",
                "overtime_sudden_death": "in overtime (sudden death)",
                "time_up": None,  # decided by crowns; shown via the score line
            }
            info = self._last_result_info if isinstance(self._last_result_info, dict) else {}
            winner_name = {"player": "Blue", "opponent": "Red"}.get(info.get("winner"), "")
            reason = reason_map.get(info.get("reason"))
            parts = []
            if info.get("reason") == "time_up":
                parts.append(f"time up -- crowns {info.get('player_crowns', eng.player_trophies)}-"
                             f"{info.get('opponent_crowns', eng.opponent_trophies)}")
            elif reason:
                parts.append(reason)
            if winner_name and info.get("winner") != "tie":
                msg = f"match over -- {winner_name} wins" + (f", {' '.join(parts)}" if parts else "")
            elif winner_name == "tie":
                msg = "match over -- draw" + (f", {' '.join(parts)}" if parts else "")
            else:  # no info captured (e.g. truncated without a result): fall back to crowns
                msg = ("match over -- crowns "
                       f"{eng.player_trophies}-{eng.opponent_trophies}")
            surf = self.font_big.render(msg, True, C_KING_RING)
            screen.blit(surf, (WIN_W // 2 - surf.get_width() // 2, WIN_H // 2))

        pg.display.flip()
        return screen.copy()

    def _hp_bar(self, cx: int, y: int, width: int, frac: float):
        """A tiny HP bar centred under/over a unit."""
        frac = max(0.0, min(1.0, frac))
        x0 = cx - width // 2
        self.pg.draw.rect(self.screen, (40, 44, 56), (x0, y, width, 4))
        color = C_READY if frac > 0.3 else (255, 180, 60) if frac > 0.15 else (255, 70, 70)
        self.pg.draw.rect(self.screen, color, (x0, y, int(width * frac), 4))

    def _draw_hud(self, own_elixir: float, foe_elixir: float, elixir_max: float,
                  hand: List[str], cooldowns: List[int], side: str):
        """Elixir bar + four card slots for one player."""
        pg = self.pg
        y0 = HUD_BOTTOM - 30 if side == "bottom" else 6
        # Elixir bar.
        bw = int(240 * (own_elixir / elixir_max))
        x0 = ARENA_X + 8
        pg.draw.rect(self.screen, C_DIM, (x0, y0, 240, 10), width=1)
        color = C_PLAYER if side == "bottom" else C_OPPONENT
        pg.draw.rect(self.screen, color, (x0, y0, bw, 10))

        # Hand slots: ready cards glow green-edged; cooling-down ones are dim.
        from ..env.sim.entities import CARD_DEFS
        for i in range(4):
            sx = x0 + 260 + i * 58
            pg.draw.rect(self.screen, (30, 34, 46), (sx, y0 - 14, 52, 38))
            if i >= len(hand) or hand[i] is None:
                continue
            card = CARD_DEFS.get(hand[i])
            cost = getattr(card, "cost", "?")
            ready = (i < len(cooldowns) and cooldowns[i] <= 0
                     and own_elixir >= float(getattr(card, "cost", 99)))
            edge = C_READY if ready else C_COOLDOWN
            pg.draw.rect(self.screen, edge, (sx, y0 - 14, 52, 38), width=2)
            label = f"{str(cost)}"
            surf = self.font_small.render(label, True, C_TEXT)
            if hasattr(surf, "get_width"):
                screen_w = WIN_W
                cx_slot = sx + (52 - surf.get_width()) // 2
                cy_slot = y0 + 3
                # Clamp so a top-side slot never draws off-screen.
                self.screen.blit(surf, (cx_slot, max(1, min(cy_slot, WIN_H - 20))))

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self, frames: Optional[int] = None) -> int:
        """Run the window until quit (or ``frames`` ticks if given). Returns exit code."""
        pg = self.pg
        clock = pg.time.Clock()
        acc = 0.0          # accumulated time; whole engine ticks are stepped out of it
        n_frames = 0
        while True:
            for event in pg.event.get():      # pump every frame -- QUIT never arrives otherwise
                if event.type == pg.QUIT:
                    return 0
                if event.type == pg.KEYDOWN:
                    key = event.key
                    if key in (pg.K_q, pg.K_ESCAPE):
                        self.pg.quit()
                        return 0
                    elif key == pg.K_SPACE:
                        self.paused = not self.paused
                    elif key == pg.K_r:
                        self.new_match()
                    elif key in (pg.K_PLUS, pg.K_EQUALS, pg.K_UP):
                        self.speed *= 1.5
                    elif key in (pg.K_MINUS, pg.K_DOWN):
                        self.speed /= 1.5

            dt = clock.tick(60) / 1000.0      # capped at 60 fps; returns elapsed seconds
            if not self.paused:
                acc += dt * self.speed        # ticks worth of sim time this frame
                while acc >= 1.0 and self.step():
                    acc -= 1.0

            self.render_frame()
            n_frames += 1
            if frames is not None and n_frames >= frames:
                break
            # Headless runs have nobody to watch the "match over" screen; stop
            # once it ends. Windowed mode keeps showing it (R restarts).
            match_over = self.engine.terminated or self.engine.truncated
            if match_over and os.environ.get("SDL_VIDEODRIVER") == "dummy":
                break

        self.pg.quit()
        return 0


def run_arena(model_a: Optional[str] = None, model_b: Optional[str] = None,
              profile_a: str = "random", profile_b: str = "balanced",
              speed: float = 10.0, headless: bool = False,
              frames: Optional[int] = None, match_seed: Optional[int] = None) -> int:
    """Entry point used by ``crp watch`` and the tests."""
    win = ArenaWindow.from_models(model_a, model_b, profile_a, profile_b,
                                  speed=speed, headless=headless or (frames is not None),
                                  match_seed=match_seed)
    return win.run(frames=frames)
