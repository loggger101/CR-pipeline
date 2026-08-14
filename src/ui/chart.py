"""Matplotlib chart embedded in a Tk frame.

Matplotlib is already a dependency and its Tk backend needs no extra install,
so it is the cheapest way to get a live line chart into the window.
"""

from __future__ import annotations

import tkinter as tk
from typing import Dict, Sequence

import matplotlib
matplotlib.use("Agg")  # Chosen before the Tk backend imports a display

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

BACKGROUND = "#1b1f27"
FOREGROUND = "#e6e9ef"
GRID = "#2b323d"
SERIES_COLOURS = ["#4da3ff", "#ffd166", "#7bd389", "#ff6b6b", "#c792ea"]


class LineChart(tk.Frame):
    """A dark-themed multi-series line chart that can be updated in place."""

    def __init__(self, master, title: str = "", xlabel: str = "",
                 ylabel: str = "", height: int = 3, **kwargs):
        kwargs.setdefault("background", BACKGROUND)
        super().__init__(master, **kwargs)

        self.figure = Figure(figsize=(5, height), dpi=100,
                             facecolor=BACKGROUND)
        self.axes = self.figure.add_subplot(111)
        self._style(title, xlabel, ylabel)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel

    def _style(self, title: str, xlabel: str, ylabel: str) -> None:
        axes = self.axes
        axes.set_facecolor(BACKGROUND)
        axes.tick_params(colors=FOREGROUND, labelsize=8)
        for spine in axes.spines.values():
            spine.set_color(GRID)
        axes.grid(True, color=GRID, linewidth=0.5)
        if title:
            axes.set_title(title, color=FOREGROUND, fontsize=10)
        if xlabel:
            axes.set_xlabel(xlabel, color=FOREGROUND, fontsize=9)
        if ylabel:
            axes.set_ylabel(ylabel, color=FOREGROUND, fontsize=9)

    def plot(self, series: Dict[str, Sequence[float]],
             x: Sequence[float] | None = None) -> None:
        """Replace the chart contents with ``series`` (name -> values)."""
        self.axes.clear()
        self._style(self._title, self._xlabel, self._ylabel)

        for index, (name, values) in enumerate(series.items()):
            if not len(values):
                continue
            xs = x if x is not None else range(1, len(values) + 1)
            self.axes.plot(list(xs), list(values), label=name, linewidth=1.6,
                           color=SERIES_COLOURS[index % len(SERIES_COLOURS)])

        if any(len(v) for v in series.values()):
            legend = self.axes.legend(loc="best", fontsize=8,
                                      facecolor=BACKGROUND,
                                      edgecolor=GRID)
            for text in legend.get_texts():
                text.set_color(FOREGROUND)

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def clear(self) -> None:
        self.plot({})
