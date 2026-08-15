"""CR-Pipeline desktop application.

A Tk window for driving the pipeline: configure and run training, watch agents
play, browse past runs, and evaluate saved agents.

Everything slow happens on a worker thread via :mod:`src.ui.jobs`; the UI drains
its event queue on a timer, so widget access stays on the main thread and the
window never blocks.
"""

from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

from .arena_canvas import ArenaCanvas
from .chart import LineChart
from .jobs import JobRunner
from .operations import (
    MATCH_DURATIONS, SCRIPTED_OPPONENTS, START_CONTINUE, START_FRESH,
    START_MODES, START_SEED, TOURNAMENT_FORMATS, build_training_config,
    default_worker_count, evaluate_against_scripted, evaluate_head_to_head,
    list_runs, load_agent_genome, load_run_history, new_run_dir, play_match,
    resolve_runs_dir, run_dir_for_resume, run_training,
)

BACKGROUND = "#161a21"
PANEL = "#1b1f27"
TEXT = "#e6e9ef"
MUTED = "#8b94a3"
ACCENT = "#4da3ff"
POLL_MS = 120


class CRPipelineApp(tk.Tk):
    """Main window."""

    def __init__(self, project_root: Optional[str] = None,
                 runs_dir: Optional[str] = None):
        super().__init__()
        self.project_root = Path(project_root or os.getcwd())
        self.runs_dir = resolve_runs_dir(str(self.project_root), runs_dir)

        self.title("CR-Pipeline")
        self.geometry("1120x760")
        self.minsize(940, 640)
        self.configure(background=BACKGROUND)

        self.jobs = JobRunner()
        self._apply_theme()
        self._build_layout()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_MS, self._pump)

    # -- chrome ------------------------------------------------------------

    def _apply_theme(self) -> None:
        style = ttk.Style(self)
        # 'clam' is the one built-in theme that honours custom colours on
        # Windows; the native themes ignore most background settings.
        style.theme_use("clam")
        style.configure(".", background=PANEL, foreground=TEXT,
                        fieldbackground=BACKGROUND, bordercolor="#2b323d")
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", background=BACKGROUND,
                        foreground=MUTED, padding=(16, 8))
        style.map("TNotebook.Tab",
                  background=[("selected", PANEL)],
                  foreground=[("selected", TEXT)])
        style.configure("TButton", background="#2b323d", foreground=TEXT,
                        padding=(12, 6), borderwidth=0)
        style.map("TButton",
                  background=[("active", "#37404e"), ("disabled", "#232833")],
                  foreground=[("disabled", MUTED)])
        style.configure("Accent.TButton", background=ACCENT,
                        foreground="#0b1220")
        style.map("Accent.TButton", background=[("active", "#6cb6ff")])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, borderwidth=0, rowheight=24)
        style.configure("Treeview.Heading", background="#232833",
                        foreground=MUTED)
        style.map("Treeview", background=[("selected", "#2f4a6b")])
        style.configure("TEntry", insertcolor=TEXT)

    def _build_layout(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 4))

        self.training_tab = TrainingTab(self.notebook, self)
        self.watch_tab = WatchTab(self.notebook, self)
        self.runs_tab = RunsTab(self.notebook, self)
        self.agents_tab = AgentsTab(self.notebook, self)

        self.notebook.add(self.training_tab, text="Train")
        self.notebook.add(self.watch_tab, text="Watch")
        self.notebook.add(self.runs_tab, text="Runs")
        self.notebook.add(self.agents_tab, text="Agents")

        self.status = tk.StringVar(value="Ready")
        bar = tk.Frame(self, background=BACKGROUND)
        bar.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(bar, textvariable=self.status, background=BACKGROUND,
                 foreground=MUTED, anchor="w").pack(side="left")
        tk.Label(bar, text=f"runs: {self.runs_dir}", background=BACKGROUND,
                 foreground="#5b6472", anchor="e").pack(side="right")

    # -- job plumbing ------------------------------------------------------

    def set_status(self, message: str) -> None:
        self.status.set(message)

    def start_job(self, name: str, target, owner) -> bool:
        """Start a background job, refusing if one is already running."""
        if self.jobs.busy:
            messagebox.showinfo(
                "Busy",
                f"A {self.jobs.current_job} job is already running. "
                f"Stop it before starting another.",
            )
            return False
        self._job_owner = owner
        started = self.jobs.start(name, target)
        if started:
            self.set_status(f"{name}: running...")
        return started

    def _pump(self) -> None:
        """Drain job events into the owning tab. Runs on the Tk event loop."""
        owner = getattr(self, "_job_owner", None)
        for event in self.jobs.drain():
            if owner is None:
                continue
            try:
                owner.handle_event(event)
            except Exception as exc:  # a bad handler must not kill the loop
                self.set_status(f"UI error: {exc}")
        self.after(POLL_MS, self._pump)

    def _on_close(self) -> None:
        if self.jobs.busy:
            if not messagebox.askokcancel(
                    "Quit", f"A {self.jobs.current_job} job is running. "
                            f"Stop it and quit?"):
                return
            self.jobs.cancel()
            self.jobs.join(timeout=10)
        self.destroy()


# ---------------------------------------------------------------------------
# Shared widgets
# ---------------------------------------------------------------------------

class Field(tk.Frame):
    """A labelled entry or combobox."""

    def __init__(self, master, label: str, default: Any = "",
                 choices: Optional[List[str]] = None, width: int = 10):
        super().__init__(master, background=PANEL)
        self.label = tk.Label(self, text=label, background=PANEL,
                              foreground=MUTED, anchor="w", width=18)
        self.label.pack(side="left")
        self.var = tk.StringVar(value=str(default))
        if choices:
            widget = ttk.Combobox(self, textvariable=self.var, values=choices,
                                  state="readonly", width=width + 4)
        else:
            widget = ttk.Entry(self, textvariable=self.var, width=width)
        widget.pack(side="left")
        self.widget = widget

    def get(self) -> str:
        return self.var.get()

    def set_label(self, text: str) -> None:
        self.label.configure(text=text)


class RunPicker(tk.Toplevel):
    """Small chooser listing runs that have a checkpoint to continue from."""

    def __init__(self, master, runs: List[dict], target: tk.StringVar):
        super().__init__(master)
        self.title("Continue a run")
        self.configure(background=PANEL)
        self.geometry("620x320")
        self.transient(master.winfo_toplevel())
        self.target = target
        self.runs = runs

        tk.Label(self, text="Pick the run to carry on training",
                 background=PANEL, foreground=TEXT,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(12, 6))

        columns = ("run", "generations", "best")
        self.tree = ttk.Treeview(self, columns=columns, show="headings",
                                 height=9)
        for column, heading, width in [
            ("run", "Run", 330), ("generations", "Generations", 110),
            ("best", "Best fitness", 120),
        ]:
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=12)

        for run in runs:
            best = ("-" if run["best_fitness"] is None
                    else f"{run['best_fitness']:.3f}")
            self.tree.insert("", "end", iid=run["path"],
                             values=(run["run_id"], run["generations"], best))

        buttons = tk.Frame(self, background=PANEL)
        buttons.pack(fill="x", padx=12, pady=10)
        ttk.Button(buttons, text="Continue this run", style="Accent.TButton",
                   command=self._choose).pack(side="left")
        ttk.Button(buttons, text="Browse instead...",
                   command=self._browse).pack(side="left", padx=6)
        ttk.Button(buttons, text="Cancel",
                   command=self.destroy).pack(side="right")
        self.tree.bind("<Double-1>", lambda _e: self._choose())

    def _choose(self) -> None:
        selection = self.tree.selection()
        if selection:
            self.target.set(selection[0])
            self.destroy()

    def _browse(self) -> None:
        path = filedialog.askdirectory(title="Select a run directory")
        if path:
            self.target.set(path)
            self.destroy()


class LogPane(tk.Text):
    """Append-only, read-only log view."""

    def __init__(self, master, height: int = 8):
        super().__init__(master, height=height, background="#12151b",
                         foreground="#c7cedb", insertbackground=TEXT,
                         borderwidth=0, highlightthickness=0,
                         font=("Consolas", 9), wrap="word")
        self.configure(state="disabled")

    def append(self, message: str) -> None:
        self.configure(state="normal")
        self.insert("end", message.rstrip() + "\n")
        self.see("end")
        self.configure(state="disabled")

    def clear(self) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

class TrainingTab(tk.Frame):
    """Configure, launch and monitor a training run."""

    def __init__(self, master, app: CRPipelineApp):
        super().__init__(master, background=PANEL)
        self.app = app
        self.history: Dict[str, List[float]] = {"best": [], "mean": []}
        self.elo_series: List[float] = []
        self.hof_series: List[float] = []

        left = tk.Frame(self, background=PANEL)
        left.pack(side="left", fill="y", padx=(12, 8), pady=12)

        tk.Label(left, text="Training", background=PANEL, foreground=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))

        # -- where the population comes from -------------------------------
        self.start_mode = tk.StringVar(value=START_FRESH)
        mode_row = tk.Frame(left, background=PANEL)
        mode_row.pack(anchor="w", pady=(0, 4))
        tk.Label(mode_row, text="Start from", background=PANEL,
                 foreground=MUTED, anchor="w", width=18).pack(side="left")
        ttk.Combobox(mode_row, textvariable=self.start_mode, values=START_MODES,
                     state="readonly", width=24).pack(side="left")
        self.start_mode.trace_add("write", lambda *_: self._toggle_start_mode())

        self.source_frame = tk.Frame(left, background=PANEL)
        self.source_frame.pack(anchor="w", fill="x", pady=(0, 4))

        self.resume_run = tk.StringVar(value="")
        self.resume_row = tk.Frame(self.source_frame, background=PANEL)
        ttk.Button(self.resume_row, text="Pick run...",
                   command=self._choose_run).pack(side="left")
        tk.Label(self.resume_row, textvariable=self.resume_run, background=PANEL,
                 foreground=ACCENT, anchor="w", width=30).pack(side="left", padx=6)

        self.seed_agents: List[str] = []
        self.seed_summary = tk.StringVar(value="none chosen")
        self.seed_row = tk.Frame(self.source_frame, background=PANEL)
        ttk.Button(self.seed_row, text="Pick agents...",
                   command=self._choose_seed_agents).pack(side="left")
        tk.Label(self.seed_row, textvariable=self.seed_summary, background=PANEL,
                 foreground=ACCENT, anchor="w", width=30).pack(side="left", padx=6)

        self.fields: Dict[str, Field] = {}
        for key, label, default, choices in [
            ("population_size", "Population", 24, None),
            ("max_generations", "Generations", 20, None),
            ("elite_count", "Elite count", 4, None),
            ("num_workers", "Workers", default_worker_count(), None),
            ("seed", "Seed", 42, None),
            ("tournament_format", "Format", "swiss", TOURNAMENT_FORMATS),
            ("tournament_matches", "Matches / pairing", 2, None),
            ("hall_of_fame_size", "Hall of fame", 4, None),
            ("match_duration", "Match length", "short", MATCH_DURATIONS),
        ]:
            field = Field(left, label, default, choices)
            field.pack(anchor="w", pady=2)
            self.fields[key] = field

        self.tournament_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Train by tournament (agents play each other)",
                        variable=self.tournament_mode,
                        command=self._toggle_mode).pack(anchor="w", pady=(10, 2))

        self.scripted_frame = tk.Frame(left, background=PANEL)
        self.scripted_frame.pack(anchor="w")
        for key, label, default, choices in [
            ("opponent_type", "Opponent", "balanced", SCRIPTED_OPPONENTS),
            ("matches_per_agent", "Matches / agent", 3, None),
        ]:
            field = Field(self.scripted_frame, label, default, choices)
            field.pack(anchor="w", pady=2)
            self.fields[key] = field
        self._toggle_mode()
        self._toggle_start_mode()

        buttons = tk.Frame(left, background=PANEL)
        buttons.pack(anchor="w", pady=(14, 0))
        self.start_button = ttk.Button(buttons, text="Start", style="Accent.TButton",
                                       command=self._start)
        self.start_button.pack(side="left", padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop,
                                      state="disabled")
        self.stop_button.pack(side="left")

        right = tk.Frame(self, background=PANEL)
        right.pack(side="left", fill="both", expand=True, padx=(8, 12), pady=12)

        self.headline = tk.StringVar(value="Not running")
        tk.Label(right, textvariable=self.headline, background=PANEL,
                 foreground=TEXT, font=("Segoe UI", 11), anchor="w").pack(
            fill="x", pady=(0, 6))

        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 8))

        self.chart = LineChart(right, title="Fitness by generation",
                               xlabel="generation", ylabel="fitness", height=3)
        self.chart.pack(fill="both", expand=True)

        self.log = LogPane(right, height=8)
        self.log.pack(fill="x", pady=(8, 0))

    def _toggle_mode(self) -> None:
        """Scripted-opponent fields only apply when not training by tournament."""
        state = "disabled" if self.tournament_mode.get() else "normal"
        for key in ("opponent_type", "matches_per_agent"):
            widget = self.fields[key].widget
            widget.configure(state="readonly" if (
                state == "normal" and isinstance(widget, ttk.Combobox)) else state)

    def _toggle_start_mode(self) -> None:
        """Show the picker that matches the chosen start mode."""
        mode = self.start_mode.get()
        self.resume_row.pack_forget()
        self.seed_row.pack_forget()
        if mode == START_CONTINUE:
            self.resume_row.pack(anchor="w", fill="x")
        elif mode == START_SEED:
            self.seed_row.pack(anchor="w", fill="x")

        # Continuing trains N *more* generations; the label says which.
        label = ("Extra generations" if mode == START_CONTINUE
                 else "Generations")
        self.fields["max_generations"].set_label(label)
        # Population size is fixed by the checkpoint when continuing.
        self.fields["population_size"].widget.configure(
            state="disabled" if mode == START_CONTINUE else "normal")

    def _choose_run(self) -> None:
        """Pick a previous run to continue."""
        runs = [r for r in list_runs(self.app.runs_dir) if r["generations"]]
        if not runs:
            path = filedialog.askdirectory(
                title="Select a run directory to continue",
                initialdir=self.app.runs_dir)
            if path:
                self.resume_run.set(path)
            return
        RunPicker(self, runs, self.resume_run)

    def _choose_seed_agents(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select agent checkpoints to start from",
            initialdir=self.app.runs_dir,
            filetypes=[("Agent checkpoint", "*.pt"), ("All files", "*.*")])
        if paths:
            self.seed_agents = list(paths)
            names = ", ".join(Path(p).parent.parent.name or Path(p).name
                              for p in self.seed_agents[:2])
            extra = ("" if len(self.seed_agents) <= 2
                     else f" +{len(self.seed_agents) - 2}")
            self.seed_summary.set(f"{len(self.seed_agents)}: {names}{extra}")

    def _values(self) -> Dict[str, Any]:
        values = {key: field.get() for key, field in self.fields.items()}
        values["tournament_mode"] = self.tournament_mode.get()
        values["start_mode"] = self.start_mode.get()
        values["resume_from"] = self.resume_run.get()
        values["seed_agents"] = list(self.seed_agents)
        return values

    def _start(self) -> None:
        values = self._values()
        continuing = values["start_mode"] == START_CONTINUE
        try:
            # Continuing writes back into the same run so its history and
            # checkpoints stay in one place; anything else gets a new folder.
            run_dir = (run_dir_for_resume(values["resume_from"]) if continuing
                       and values["resume_from"]
                       else new_run_dir(self.app.runs_dir))
            config = build_training_config(values, run_dir)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Check the settings", str(exc))
            return

        self.history = {"best": [], "mean": []}
        self.elo_series = []
        self.hof_series = []
        self.chart.clear()
        self.log.clear()
        self.progress.configure(maximum=config.max_generations, value=0)
        if continuing:
            self.log.append(f"Continuing {Path(run_dir).name} for "
                            f"{config.additional_generations} more generations.")
        elif config.seed_agents:
            self.log.append(f"Seeding population from "
                            f"{len(config.seed_agents)} agent(s).")

        started = self.app.start_job(
            "training", lambda ctx: run_training(ctx, config), self)
        if started:
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.headline.set("Starting...")

    def _stop(self) -> None:
        self.app.jobs.cancel()
        self.stop_button.configure(state="disabled")
        self.headline.set("Stopping after this generation...")
        self.app.set_status("training: stopping...")

    def handle_event(self, event) -> None:
        if event.kind == "progress":
            self._on_generation(event.payload)
        elif event.kind == "log":
            self.log.append(event.payload)
        elif event.kind == "done":
            self._on_done(event.payload)
        elif event.kind == "error":
            self._reset_buttons()
            self.headline.set("Training failed")
            self.log.append(event.payload["traceback"])
            self.app.set_status("training: failed")
            messagebox.showerror("Training failed", event.payload["message"])

    def _on_generation(self, snapshot: dict) -> None:
        self.history["best"].append(snapshot["best_fitness"])
        self.history["mean"].append(snapshot["mean_fitness"])
        self.progress.configure(value=snapshot["generation"])

        champion_elo = snapshot.get("champion_elo")
        hof_elo = snapshot.get("hall_of_fame_elo")
        parts = [f"gen {snapshot['generation']}/{snapshot['total_generations']}"]
        if champion_elo is not None:
            parts.append(f"champion ELO {champion_elo:.0f}")
            self.elo_series.append(champion_elo)
        if hof_elo is not None:
            parts.append(f"past champions {hof_elo:.0f}")
            self.hof_series.append(hof_elo)
        parts.append(f"fitness {snapshot['best_fitness']:.3f}")
        if snapshot.get("champion_record"):
            parts.append(snapshot["champion_record"])
        self.headline.set("   ".join(parts))

        # In tournament mode the fitness curve is points-per-match inside a
        # closed field, so its mean is pinned near 0.5 however good the
        # population gets. Ratings against the hall of fame are what show
        # progress, so chart those when they exist.
        if self.elo_series:
            series = {"champion ELO": self.elo_series}
            if self.hof_series:
                series["past champions"] = self.hof_series
            self.chart.set_labels("Rating by generation", "generation", "ELO")
        else:
            series = dict(self.history)
            self.chart.set_labels("Fitness by generation", "generation",
                                  "fitness")
        self.chart.plot(series)
        self.app.set_status(
            f"training: generation {snapshot['generation']}"
            f"/{snapshot['total_generations']}")

    def _on_done(self, result: dict) -> None:
        self._reset_buttons()
        verb = "stopped" if result.get("cancelled") else "finished"
        label = result.get("best_score_label", "fitness")
        score = result.get("best_score", 0.0)
        self.headline.set(
            f"Training {verb}: {result['generations_completed']} generations, "
            f"best {label} {score:.1f} "
            f"(gen {result['best_generation']}) in {result['elapsed']:.0f}s")
        self.log.append(f"Training {verb}.")
        self.log.append(f"Run directory: {result.get('run_dir', '?')}")
        if result.get("best_agent_path"):
            self.log.append(f"Best agent saved to {result['best_agent_path']}")
            self.app.agents_tab.suggest_agent(result["best_agent_path"])
            self.app.watch_tab.suggest_agent(result["best_agent_path"])
        self.app.runs_tab.refresh()
        self.app.set_status(f"training {verb}")

    def _reset_buttons(self) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")


class WatchTab(tk.Frame):
    """Watch an agent play a match on the arena."""

    def __init__(self, master, app: CRPipelineApp):
        super().__init__(master, background=PANEL)
        self.app = app
        self.recording = None
        self.frame_index = 0
        self._playing = False
        self._after_id = None

        controls = tk.Frame(self, background=PANEL)
        controls.pack(fill="x", padx=12, pady=(12, 6))

        self.agent_path = tk.StringVar(value="")
        ttk.Button(controls, text="Load agent...",
                   command=self._choose_agent).pack(side="left")
        tk.Label(controls, textvariable=self.agent_path, background=PANEL,
                 foreground=MUTED, anchor="w").pack(
            side="left", padx=8, fill="x", expand=True)

        self.opponent = Field(controls, "Opponent", "balanced", SCRIPTED_OPPONENTS)
        self.opponent.pack(side="left", padx=(0, 8))
        self.seed = Field(controls, "Seed", 7, width=6)
        self.seed.pack(side="left", padx=(0, 8))
        self.play_button = ttk.Button(controls, text="Play match",
                                      style="Accent.TButton", command=self._play)
        self.play_button.pack(side="left")

        self.arena = ArenaCanvas(self, width=760, height=560)
        self.arena.pack(fill="both", expand=True, padx=12, pady=6)

        transport = tk.Frame(self, background=PANEL)
        transport.pack(fill="x", padx=12, pady=(0, 12))
        self.playpause = ttk.Button(transport, text="Pause",
                                    command=self._toggle_play, state="disabled")
        self.playpause.pack(side="left", padx=(0, 8))
        self.scrub = ttk.Scale(transport, from_=0, to=0, orient="horizontal",
                               command=self._scrub)
        self.scrub.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.speed = Field(transport, "Speed (fps)", 30, width=5)
        self.speed.pack(side="left")
        self.outcome = tk.StringVar(value="")
        tk.Label(self, textvariable=self.outcome, background=PANEL,
                 foreground=TEXT, anchor="w").pack(fill="x", padx=12,
                                                   pady=(0, 10))

    def suggest_agent(self, path: str) -> None:
        if not self.agent_path.get():
            self.agent_path.set(path)

    def _choose_agent(self) -> None:
        path = filedialog.askopenfilename(
            title="Select an agent checkpoint",
            initialdir=self.app.runs_dir,
            filetypes=[("Agent checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.agent_path.set(path)

    def _play(self) -> None:
        path = self.agent_path.get()
        if not path:
            messagebox.showinfo("Pick an agent",
                                "Load an agent checkpoint first.")
            return
        try:
            genome = load_agent_genome(path)
            seed = int(self.seed.get())
        except (ValueError, OSError) as exc:
            messagebox.showerror("Could not load agent", str(exc))
            return

        opponent = self.opponent.get()
        self._stop_playback()
        self.outcome.set("Simulating...")
        self.app.start_job(
            "match",
            lambda ctx: play_match(ctx, genome, opponent=opponent, seed=seed),
            self)

    def handle_event(self, event) -> None:
        if event.kind == "done":
            self.recording = event.payload
            self.frame_index = 0
            self.scrub.configure(to=max(0, len(self.recording) - 1))
            self.outcome.set(
                f"{self.recording.winner} wins by {self.recording.reason} "
                f"after {self.recording.ticks} ticks "
                f"({self.recording.ticks / 10:.0f}s) - "
                f"{len(self.recording)} frames")
            self.playpause.configure(state="normal")
            self.app.set_status("match ready")
            self._playing = True
            self._advance()
        elif event.kind == "error":
            self.outcome.set("Match failed")
            self.app.set_status("match: failed")
            messagebox.showerror("Match failed", event.payload["message"])

    def _toggle_play(self) -> None:
        self._playing = not self._playing
        self.playpause.configure(text="Pause" if self._playing else "Play")
        if self._playing:
            self._advance()

    def _stop_playback(self) -> None:
        self._playing = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _advance(self) -> None:
        if not self._playing or self.recording is None:
            return
        if self.frame_index >= len(self.recording):
            self._playing = False
            self.playpause.configure(text="Play")
            return
        self._show_frame(self.frame_index)
        self.scrub.set(self.frame_index)
        self.frame_index += 1
        try:
            fps = max(1, min(120, int(self.speed.get())))
        except ValueError:
            fps = 30
        self._after_id = self.after(int(1000 / fps), self._advance)

    def _scrub(self, value) -> None:
        if self.recording is None:
            return
        self._playing = False
        self.playpause.configure(text="Play")
        index = int(float(value))
        self.frame_index = index
        self._show_frame(index)

    def _show_frame(self, index: int) -> None:
        if self.recording and 0 <= index < len(self.recording):
            self.arena.show(self.recording.frames[index])


class RunsTab(tk.Frame):
    """Browse past runs and compare their fitness curves."""

    def __init__(self, master, app: CRPipelineApp):
        super().__init__(master, background=PANEL)
        self.app = app

        header = tk.Frame(self, background=PANEL)
        header.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(header, text="Training runs", background=PANEL,
                 foreground=TEXT, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(header, text="Refresh", command=self.refresh).pack(side="right")

        columns = ("run_id", "generations", "best_fitness", "agent")
        self.tree = ttk.Treeview(self, columns=columns, show="headings",
                                 selectmode="extended", height=9)
        for column, heading, width in [
            ("run_id", "Run", 340), ("generations", "Generations", 110),
            ("best_fitness", "Best fitness", 120), ("agent", "Saved agent", 110),
        ]:
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="x", padx=12)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._plot_selected())

        self.chart = LineChart(self, title="Best fitness by generation",
                               xlabel="generation", ylabel="fitness", height=3)
        self.chart.pack(fill="both", expand=True, padx=12, pady=(10, 12))

        self._runs: List[dict] = []
        self.refresh()

    def refresh(self) -> None:
        self._runs = list_runs(self.app.runs_dir)
        self.tree.delete(*self.tree.get_children())
        for run in self._runs:
            best = ("-" if run["best_fitness"] is None
                    else f"{run['best_fitness']:.3f}")
            self.tree.insert("", "end", iid=run["path"], values=(
                run["run_id"], run["generations"], best,
                "yes" if run["has_agent"] else "-"))
        if not self._runs:
            self.chart.clear()

    def _plot_selected(self) -> None:
        series = {}
        for path in self.tree.selection():
            history = load_run_history(path)
            if history.get("best"):
                series[Path(path).name[:28]] = history["best"]
        self.chart.plot(series)

    def handle_event(self, event) -> None:  # no background jobs here
        pass


class AgentsTab(tk.Frame):
    """Evaluate a saved agent against baselines or another agent."""

    def __init__(self, master, app: CRPipelineApp):
        super().__init__(master, background=PANEL)
        self.app = app

        picker = tk.Frame(self, background=PANEL)
        picker.pack(fill="x", padx=12, pady=(12, 6))

        self.agent_a = tk.StringVar(value="")
        self.agent_b = tk.StringVar(value="")
        for label, var in (("Agent A", self.agent_a), ("Agent B", self.agent_b)):
            row = tk.Frame(picker, background=PANEL)
            row.pack(fill="x", pady=2)
            ttk.Button(row, text=f"Load {label}...",
                       command=lambda v=var: self._choose(v)).pack(side="left")
            tk.Label(row, textvariable=var, background=PANEL, foreground=MUTED,
                     anchor="w").pack(side="left", padx=8, fill="x", expand=True)

        actions = tk.Frame(self, background=PANEL)
        actions.pack(fill="x", padx=12, pady=(6, 6))
        self.matches = Field(actions, "Matches", 6, width=6)
        self.matches.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Test A vs baselines", style="Accent.TButton",
                   command=self._vs_baselines).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="A vs B head-to-head",
                   command=self._head_to_head).pack(side="left")

        columns = ("opponent", "record", "win_rate", "towers")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=8)
        for column, heading, width in [
            ("opponent", "Opponent", 220), ("record", "W/D/L", 120),
            ("win_rate", "Win rate", 110), ("towers", "Avg towers", 110),
        ]:
            self.tree.heading(column, text=heading)
            self.tree.column(column, width=width, anchor="w")
        self.tree.pack(fill="x", padx=12, pady=(6, 6))

        self.log = LogPane(self, height=10)
        self.log.pack(fill="both", expand=True, padx=12, pady=(6, 12))

    def suggest_agent(self, path: str) -> None:
        if not self.agent_a.get():
            self.agent_a.set(path)

    def _choose(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select an agent checkpoint",
            initialdir=self.app.runs_dir,
            filetypes=[("Agent checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _load(self, var: tk.StringVar, label: str):
        path = var.get()
        if not path:
            messagebox.showinfo("Pick an agent", f"Load {label} first.")
            return None
        try:
            return load_agent_genome(path)
        except (ValueError, OSError) as exc:
            messagebox.showerror(f"Could not load {label}", str(exc))
            return None

    def _matches(self) -> int:
        try:
            return max(1, int(self.matches.get()))
        except ValueError:
            return 6

    def _vs_baselines(self) -> None:
        genome = self._load(self.agent_a, "Agent A")
        if genome is None:
            return
        self.tree.delete(*self.tree.get_children())
        self.log.clear()
        matches = self._matches()
        self.app.start_job(
            "evaluation",
            lambda ctx: evaluate_against_scripted(
                ctx, genome, SCRIPTED_OPPONENTS, matches=matches),
            self)

    def _head_to_head(self) -> None:
        genome_a = self._load(self.agent_a, "Agent A")
        genome_b = self._load(self.agent_b, "Agent B")
        if genome_a is None or genome_b is None:
            return
        self.tree.delete(*self.tree.get_children())
        self.log.clear()
        matches = self._matches()
        self.app.start_job(
            "evaluation",
            lambda ctx: evaluate_head_to_head(ctx, genome_a, genome_b,
                                              matches=matches),
            self)

    def handle_event(self, event) -> None:
        if event.kind == "log":
            self.log.append(event.payload)
        elif event.kind == "progress":
            self._add_row(event.payload)
        elif event.kind == "done":
            payload = event.payload
            if isinstance(payload, dict) and "a_win_rate" in payload:
                self.log.append(
                    f"Head-to-head over {payload['matches']} matches: "
                    f"A {payload['a_wins']}W/{payload['a_draws']}D/"
                    f"{payload['a_losses']}L  ({payload['a_win_rate']:.0%})")
                self.tree.insert("", "end", values=(
                    "Agent B (head-to-head)",
                    f"{payload['a_wins']}/{payload['a_draws']}/{payload['a_losses']}",
                    f"{payload['a_win_rate']:.0%}", "-"))
            self.app.set_status("evaluation finished")
        elif event.kind == "error":
            self.log.append(event.payload["traceback"])
            self.app.set_status("evaluation: failed")
            messagebox.showerror("Evaluation failed", event.payload["message"])

    def _add_row(self, row: dict) -> None:
        self.tree.insert("", "end", values=(
            row["opponent"],
            f"{row['wins']}/{row['draws']}/{row['losses']}",
            f"{row['win_rate']:.0%}",
            f"{row['avg_towers']:.1f}"))


def launch(project_root: Optional[str] = None,
           runs_dir: Optional[str] = None) -> None:
    """Open the application window."""
    CRPipelineApp(project_root, runs_dir).mainloop()
