"""Arena renderer drawn straight onto a Tk canvas.

``src/viz/rendering.py`` produces RGB numpy arrays, which would need Pillow to
reach a Tk widget. Drawing canvas items directly avoids that dependency, keeps
redraws cheap (items are moved, not re-rasterised), and lets the arena scale
with the window.
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional

BACKGROUND = "#1b1f27"
GRID = "#2b323d"
RIVER = "#1d3a52"
BRIDGE = "#5a4632"
PLAYER = "#4da3ff"
OPPONENT = "#ff6b6b"
PLAYER_TOWER = "#2f6fb5"
OPPONENT_TOWER = "#b53f3f"
KING_RING = "#ffd166"
TEXT = "#e6e9ef"
MUTED = "#8b94a3"


class ArenaCanvas(tk.Canvas):
    """Draws a simulation snapshot: towers, troops, elixir and the clock."""

    def __init__(self, master, grid_cols: int = 8, grid_rows: int = 6,
                 bridge_cols=(3, 4), bridge_row: int = 2, **kwargs):
        kwargs.setdefault("background", BACKGROUND)
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.bridge_cols = tuple(bridge_cols)
        self.bridge_row = bridge_row
        self._snapshot: Optional[dict] = None
        self.bind("<Configure>", lambda _event: self._redraw())

    # -- geometry ----------------------------------------------------------

    def _cell(self):
        """Pixel size of one grid cell, keeping the arena square-ish."""
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        return width / self.grid_cols, height / self.grid_rows

    def _to_px(self, col: float, row: float):
        cw, ch = self._cell()
        return (col + 0.5) * cw, (row + 0.5) * ch

    # -- public API --------------------------------------------------------

    def show(self, snapshot: Optional[dict]) -> None:
        """Render a snapshot produced by :func:`snapshot_from_engine`."""
        self._snapshot = snapshot
        self._redraw()

    def clear(self) -> None:
        self.show(None)

    # -- drawing -----------------------------------------------------------

    def _redraw(self) -> None:
        self.delete("all")
        self._draw_board()
        if self._snapshot is None:
            self._draw_placeholder()
            return
        self._draw_towers()
        self._draw_units()
        self._draw_hud()

    def _draw_board(self) -> None:
        cw, ch = self._cell()
        width, height = self.winfo_width(), self.winfo_height()

        # River across the bridge row, with the two crossings picked out.
        self.create_rectangle(0, self.bridge_row * ch, width,
                              (self.bridge_row + 1) * ch,
                              fill=RIVER, outline="")
        for col in self.bridge_cols:
            self.create_rectangle(col * cw, self.bridge_row * ch,
                                  (col + 1) * cw, (self.bridge_row + 1) * ch,
                                  fill=BRIDGE, outline="")

        for col in range(1, self.grid_cols):
            self.create_line(col * cw, 0, col * cw, height, fill=GRID)
        for row in range(1, self.grid_rows):
            self.create_line(0, row * ch, width, row * ch, fill=GRID)

    def _draw_placeholder(self) -> None:
        self.create_text(self.winfo_width() / 2, self.winfo_height() / 2,
                         text="No match loaded", fill=MUTED,
                         font=("Segoe UI", 11))

    def _draw_towers(self) -> None:
        cw, ch = self._cell()
        size = min(cw, ch) * 0.34
        for tower in self._snapshot.get("towers", []):
            x, y = self._to_px(tower["col"], tower["row"])
            colour = PLAYER_TOWER if tower["owner"] == "player" else OPPONENT_TOWER
            if not tower["alive"]:
                colour = "#3a3f4a"
            self.create_rectangle(x - size, y - size, x + size, y + size,
                                  fill=colour, outline=KING_RING if tower["king"] else "",
                                  width=2 if tower["king"] else 0)
            if tower["alive"] and tower["max_hp"]:
                self._hp_bar(x, y - size - 5, size * 2,
                             tower["hp"] / tower["max_hp"], colour)

    def _draw_units(self) -> None:
        cw, ch = self._cell()
        radius = min(cw, ch) * 0.16
        for unit in self._snapshot.get("units", []):
            x, y = self._to_px(unit["col"], unit["row"])
            colour = PLAYER if unit["owner"] == "player" else OPPONENT
            if unit["air"]:
                # Air units drawn as diamonds so they read apart from ground.
                self.create_polygon(x, y - radius, x + radius, y,
                                    x, y + radius, x - radius, y,
                                    fill=colour, outline="")
            else:
                self.create_oval(x - radius, y - radius, x + radius, y + radius,
                                 fill=colour, outline="")
            if unit["max_hp"]:
                self._hp_bar(x, y - radius - 4, radius * 2,
                             unit["hp"] / unit["max_hp"], colour)

    def _hp_bar(self, cx: float, cy: float, width: float,
                fraction: float, colour: str) -> None:
        fraction = max(0.0, min(1.0, fraction))
        half = width / 2
        self.create_rectangle(cx - half, cy - 2, cx + half, cy + 2,
                              fill="#11141a", outline="")
        self.create_rectangle(cx - half, cy - 2, cx - half + width * fraction,
                              cy + 2, fill=colour, outline="")

    def _draw_hud(self) -> None:
        snap = self._snapshot
        pad = 8
        self.create_text(
            pad, pad, anchor="nw", fill=TEXT, font=("Segoe UI", 9),
            text=(f"tick {snap.get('tick', 0)}   "
                  f"crowns {snap.get('player_crowns', 0)}-"
                  f"{snap.get('opponent_crowns', 0)}"
                  + ("   OVERTIME" if snap.get("overtime") else "")),
        )
        self.create_text(
            pad, self.winfo_height() - pad, anchor="sw",
            fill=PLAYER, font=("Segoe UI", 9),
            text=f"you  {snap.get('player_elixir', 0):.1f} elixir",
        )
        self.create_text(
            self.winfo_width() - pad, pad, anchor="ne",
            fill=OPPONENT, font=("Segoe UI", 9),
            text=f"{snap.get('opponent_elixir', 0):.1f} elixir  opponent",
        )
        if snap.get("winner"):
            self.create_text(
                self.winfo_width() / 2, self.winfo_height() / 2,
                fill=TEXT, font=("Segoe UI", 16, "bold"),
                text=f"{snap['winner']} wins ({snap.get('reason', '')})",
            )


def snapshot_from_engine(engine, winner: Optional[str] = None,
                         reason: Optional[str] = None) -> dict:
    """Capture the drawable state of a live engine.

    Returns plain data rather than engine objects so snapshots can be queued
    across threads and replayed without holding the simulation open.
    """
    towers = [
        {
            "col": t.col, "row": t.row, "owner": t.owner,
            "hp": t.hp, "max_hp": t.max_hp,
            "alive": t.is_alive, "king": t.is_king,
        }
        for t in list(engine.player_towers) + list(engine.opponent_towers)
    ]
    units = [
        {
            "col": u.col, "row": u.row, "owner": u.owner, "type": u.unit_type,
            "hp": u.hp, "max_hp": u.max_hp, "air": u.is_air,
        }
        for u in list(engine.player_units) + list(engine.opponent_units)
        if u.is_alive and not u.is_building
    ]
    return {
        "tick": engine.tick,
        "towers": towers,
        "units": units,
        "player_elixir": engine.player_elixir,
        "opponent_elixir": engine.opponent_elixir,
        "player_crowns": engine.player_trophies,
        "opponent_crowns": engine.opponent_trophies,
        "overtime": engine.is_overtime,
        "winner": winner,
        "reason": reason,
    }
