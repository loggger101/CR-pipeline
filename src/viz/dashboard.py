"""Streamlit dashboard for real-time training visualization.

Provides:
- Fitness curve plots
- Population statistics
- Top agent replay viewer
- Training controls (pause, resume, checkpoint)
- Hyperparameter display
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Dashboard is optional - only import if streamlit is available
try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def run_dashboard(runs_dir: str = "runs", refresh_interval: int = 5) -> None:
    """Run the training visualization dashboard.

    Args:
        runs_dir: Directory containing training runs.
        refresh_interval: Seconds between dashboard refreshes.
    """
    if not HAS_STREAMLIT:
        logger.warning("Streamlit not installed. Install with: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="CR-Pipeline Training Dashboard",
        page_icon="🏰",
        layout="wide",
    )

    st.title("🏰 CR-Pipeline Training Dashboard")
    st.markdown("Evolutionary Neural Network Training for Clash Royale")

    # Sidebar controls
    st.sidebar.header("Controls")
    runs_path = Path(runs_dir)

    # Available generations
    gen_dirs = sorted([d for d in runs_path.glob("gen_*") if d.is_dir()],
                      key=lambda x: int(x.name.split("_")[1]))

    if gen_dirs:
        gen_numbers = [int(d.name.split("_")[1]) for d in gen_dirs]
        selected_gen = st.sidebar.selectbox(
            "Select Generation",
            options=gen_numbers,
            index=len(gen_numbers) - 1,
        )
    else:
        selected_gen = None
        st.sidebar.warning("No training generations found.")

    # Training status
    if selected_gen:
        _render_generation_view(runs_path / f"gen_{selected_gen:04d}")

    # Fitness curves from best directory
    best_dir = runs_path / "best"
    if best_dir.exists():
        st.sidebar.header("Best Agent")
        meta_path = best_dir / "metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            st.sidebar.metric("Best Fitness", f"{meta.get('fitness', 0):.3f}")
            st.sidebar.metric("Generation", str(meta.get('generation', 0)))

    # Auto-refresh
    if selected_gen:
        time.sleep(refresh_interval)
        st.rerun()


def _render_generation_view(gen_dir: Path) -> None:
    """Render the view for a specific generation.

    Args:
        gen_dir: Path to the generation directory.
    """
    # Load metrics
    metrics_path = gen_dir / "metrics.json"
    if not metrics_path.exists():
        st.error("No metrics file found for this generation.")
        return

    import json
    with open(metrics_path) as f:
        metrics = json.load(f)

    # Load fitness history
    history_path = gen_dir / "fitness_history.json"
    if history_path.exists():
        with open(history_path) as f:
            fitness_history = json.load(f)
    else:
        fitness_history = {}

    # Header stats
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Best Fitness", f"{metrics.get('best_fitness', 0):.3f}")
    col2.metric("Mean Fitness", f"{metrics.get('mean_fitness', 0):.3f}")
    col3.metric("Median", f"{metrics.get('median_fitness', 0):.3f}")
    col4.metric("Std Dev", f"{metrics.get('std_fitness', 0):.3f}")
    col5.metric("Diversity", f"{metrics.get('diversity', 0):.3f}")

    # Fitness curves
    if fitness_history:
        st.subheader("Fitness Curves")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Best Fitness", "Mean Fitness",
                          "Min Fitness", "Std Deviation"),
            vertical_spacing=0.15,
        )

        gen_range = list(range(1, len(fitness_history.get("best", [1])) + 1))

        for i, (key, title, row, col) in enumerate([
            ("best", "Best Fitness", 1, 1),
            ("mean", "Mean Fitness", 1, 2),
            ("min", "Min Fitness", 2, 1),
            ("std", "Std Deviation", 2, 2),
        ]):
            values = fitness_history.get(key, [])
            if values:
                fig.add_trace(
                    go.Scatter(x=gen_range, y=values, name=title,
                              line=dict(width=2)),
                    row=row, col=col,
                )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Population statistics table
    st.subheader("Population Statistics")
    stats_data = {
        "Metric": ["Best", "Mean", "Median", "Min", "Max", "Std", "Diversity"],
        "Value": [
            metrics.get("best_fitness", 0),
            metrics.get("mean_fitness", 0),
            metrics.get("median_fitness", 0),
            metrics.get("min_fitness", 0),
            metrics.get("max_fitness", 0),
            metrics.get("std_fitness", 0),
            metrics.get("diversity", 0),
        ],
    }
    st.table(stats_data)

    # Top agents
    st.subheader("Top Agents")
    # Would load from population.pt if needed
    st.info("Top agent details available in checkpoint files.")


def render_fitness_curves(fitness_history: Dict[str, List[float]],
                          title: str = "Fitness Over Generations") -> None:
    """Render fitness curves using Plotly.

    Args:
        fitness_history: Dictionary of fitness metrics over generations.
        title: Chart title.
    """
    if not HAS_STREAMLIT:
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    generations = list(range(1, len(fitness_history.get("best", [1])) + 1))

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Best Fitness",
            "Population Statistics",
            "Diversity",
        ),
        vertical_spacing=0.12,
    )

    # Best fitness
    if fitness_history.get("best"):
        fig.add_trace(
            go.Scatter(x=generations, y=fitness_history["best"],
                      name="Best", line=dict(color="green", width=2)),
            row=1, col=1,
        )

    # Mean, median, min, max
    colors = {"mean": "blue", "median": "orange", "min": "red", "max": "purple"}
    for key, color in colors.items():
        if fitness_history.get(key):
            fig.add_trace(
                go.Scatter(x=generations, y=fitness_history[key],
                          name=key.capitalize(), line=dict(color=color, width=1.5)),
                row=2, col=1,
            )

    # Diversity
    if fitness_history.get("diversity"):
        fig.add_trace(
            go.Scatter(x=generations, y=fitness_history["diversity"],
                      name="Diversity", line=dict(color="cyan", width=1.5)),
            row=3, col=1,
        )

    fig.update_layout(height=700, title_text=title, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
