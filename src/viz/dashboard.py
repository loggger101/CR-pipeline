"""Advanced Streamlit dashboard for CR-Pipeline training visualization.

Provides:
- Multi-tab interface with comprehensive analytics
- Run comparison with statistical significance testing
- Fitness curve visualization with smoothing options
- Population statistics and distribution plots
- Tournament deep-dive with ELO, brackets, and H2H analysis
- Generation efficiency and bottleneck detection
- Configuration comparison between runs
- Export capabilities for reports and charts
"""

from __future__ import annotations

import json
import logging
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


def run_advanced_dashboard(runs_dir: str = "runs", refresh_interval: int = 10) -> None:
    """Run the advanced training visualization dashboard.

    Features:
    - Run discovery and selection
    - Multi-tab analytics interface
    - Run comparison with statistical testing
    - Tournament visualization
    - Export capabilities

    Args:
        runs_dir: Directory containing training runs.
        refresh_interval: Seconds between dashboard refreshes.
    """
    if not HAS_STREAMLIT:
        logger.warning("Streamlit not installed. Install with: pip install streamlit plotly")
        return

    st.set_page_config(
        page_title="CR-Pipeline Advanced Dashboard",
        page_icon="🏰",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🏰 CR-Pipeline Advanced Training Dashboard")
    st.markdown("Comprehensive analytics for evolutionary neural network training")

    # Initialize run manager
    from .runs_manager import RunManager
    run_manager = RunManager(runs_dir)

    # Discover runs
    runs = run_manager.discover_runs()
    fitness_data = run_manager.load_fitness_data()
    tournament_data = run_manager.load_tournament_data()
    elo_data = run_manager.load_elo_data()

    # Sidebar
    _render_sidebar(runs, run_manager, fitness_data, tournament_data, elo_data, runs_dir)


def _render_sidebar(runs: List, run_manager, fitness_data, tournament_data, elo_data, runs_dir: str) -> None:
    """Render the sidebar with run selection and controls."""
    st.sidebar.header("📊 Run Selection")

    if not runs:
        st.sidebar.warning("No runs discovered. Ensure runs are in the specified directory.")
        return

    # Run selector
    run_names = [f"{r.name} ({r.actual_generations} gens)" for r in runs]
    selected_idx = st.sidebar.selectbox(
        "Select Run",
        options=range(len(runs)),
        format_func=lambda x: run_names[x],
    )
    selected_run = runs[selected_idx] if selected_idx is not None and 0 <= selected_idx < len(runs) else None

    # Multi-run selector for comparison
    st.sidebar.header("📈 Compare Runs")
    all_run_ids = [r.run_id for r in runs]
    selected_compare = st.sidebar.multiselect(
        "Runs to Compare",
        options=all_run_ids,
        default=[selected_run.run_id] if selected_run else [],
    )

    # Run metadata display
    if selected_run:
        st.sidebar.header("ℹ️ Run Info")
        st.sidebar.metric("Best Fitness", f"{selected_run.best_fitness:.3f}")
        st.sidebar.metric("Generations", f"{selected_run.actual_generations}/{selected_run.max_generations}")
        st.sidebar.text(f"Progress: {selected_run.progress:.1%}")
        if selected_run.tournament_config:
            st.sidebar.text(f"Tournament: {selected_run.tournament_config.get('format', 'N/A')}")
        if selected_run.tags:
            st.sidebar.text(f"Tags: {', '.join(selected_run.tags)}")

    # Summary
    st.sidebar.header("📋 Summary")
    summary = run_manager.get_summary()
    st.sidebar.metric("Total Runs", summary["total_runs"])
    st.sidebar.metric("Completed", summary["completed_runs"])
    st.sidebar.metric("Running", summary["running_runs"])
    if summary["best_run"]:
        st.sidebar.text(f"Best Run: {summary['best_run']}")

    # Export options
    st.sidebar.header("💾 Export")
    if st.sidebar.button("Export Comparison"):
        if selected_compare:
            output_path = f"runs/comparison_{int(time.time())}.json"
            run_manager.export_comparison(selected_compare, output_path)
            st.sidebar.success(f"Exported to {output_path}")

    # Auto-refresh
    if selected_run:
        time.sleep(refresh_interval)
        st.rerun()


def _render_main_tabs(runs: List, run_manager, fitness_data, tournament_data, elo_data, selected_compare: List[str]) -> None:
    """Render the main content tabs."""
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Fitness", "📈 Statistics", "🏆 Tournament", "🔬 Comparison", "📦 Runs", "⚙️ Config", "📡 Monitoring", "🃏 Card Meta"
    ])

    with tab1:
        _render_fitness_tab(runs, run_manager, fitness_data, selected_compare)

    with tab2:
        _render_statistics_tab(runs, run_manager, fitness_data, selected_compare)

    with tab3:
        _render_tournament_tab(runs, run_manager, tournament_data, elo_data, selected_compare)

    with tab4:
        _render_comparison_tab(runs, run_manager, fitness_data, selected_compare)

    with tab5:
        _render_runs_tab(runs, run_manager, fitness_data, tournament_data)

    with tab6:
        _render_config_tab(runs, run_manager)

    with tab7:
        _render_monitoring_tab(runs, run_manager)

    with tab8:
        _render_card_meta_tab(runs, run_manager, fitness_data)


def _render_fitness_tab(runs: List, run_manager, fitness_data: Dict, selected_compare: List[str]) -> None:
    """Render fitness curves tab."""
    st.header("📊 Fitness Curves")

    if not selected_compare:
        st.info("Select runs to compare in the sidebar.")
        return

    # Smoothing options
    col1, col2, col3 = st.columns(3)
    with col1:
        smoothing_method = st.selectbox("Smoothing", ["None", "Moving Average", "Exponential", "Savitzky-Golay"])
    with col2:
        window_size = st.slider("Window Size", 1, 50, 10)
    with col3:
        ema_alpha = st.slider("EMA Alpha", 0.01, 0.5, 0.1, 0.01)

    # Metric selection
    metric = st.selectbox("Metric", ["best", "mean", "median", "min", "max", "std"])

    # Create figure
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i, run_id in enumerate(selected_compare):
        data = fitness_data.get(run_id, {})
        values = data.get(metric, [])

        if not values:
            continue

        generations = list(range(1, len(values) + 1))
        color = colors[i % len(colors)]

        # Apply smoothing
        if smoothing_method == "Moving Average":
            smoothed = _moving_average(values, window_size)
            fig.add_trace(go.Scatter(x=generations, y=smoothed, name=f"{run_id} (smoothed)",
                                     line=dict(color=color, width=2), opacity=0.7))
        elif smoothing_method == "Exponential":
            ema = _exponential_moving_average(values, ema_alpha)
            fig.add_trace(go.Scatter(x=generations, y=ema, name=f"{run_id} (EMA)",
                                     line=dict(color=color, width=2), opacity=0.7))
        elif smoothing_method == "Savitzky-Golay":
            smoothed = _savitzky_golay(values, window_size)
            fig.add_trace(go.Scatter(x=generations, y=smoothed, name=f"{run_id} (SG)",
                                     line=dict(color=color, width=2), opacity=0.7))
        else:
            fig.add_trace(go.Scatter(x=generations, y=values, name=run_id,
                                     line=dict(color=color, width=2)))

    fig.update_layout(
        title=f"{metric.capitalize()} Fitness Over Generations",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        height=500,
        showlegend=True,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Fitness distribution histogram
    if len(selected_compare) >= 1:
        st.subheader("Final Generation Distribution")
        col1, col2 = st.columns(2)

        for i, run_id in enumerate(selected_compare[:2]):
            data = fitness_data.get(run_id, {})
            best_values = data.get("best", [])
            mean_values = data.get("mean", [])

            with col1 if i == 0 else col2:
                st.markdown(f"### {run_id}")
                if mean_values:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=mean_values, nbinsx=30, name="Mean Fitness",
                                                     marker_color=colors[i % len(colors)]))
                    fig_hist.update_layout(
                        title="Final Generation Fitness Distribution",
                        height=300,
                        bargap=0.1,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)


def _render_statistics_tab(runs: List, run_manager, fitness_data: Dict, selected_compare: List[str]) -> None:
    """Render advanced statistics tab."""
    st.header("📈 Advanced Statistics")

    from .metrics import TrainingMetrics

    if not selected_compare:
        st.info("Select runs to analyze in the sidebar.")
        return

    # Select run to analyze
    run_id = st.selectbox("Select Run", selected_compare)
    data = fitness_data.get(run_id, {})

    if not data:
        st.warning("No fitness data available for this run.")
        return

    metrics = TrainingMetrics()
    metrics.history = {k: list(v) for k, v in data.items()}

    # Generation efficiency
    st.subheader("⚡ Generation Efficiency")
    efficiency = metrics.get_generation_efficiency()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Improvement", f"{efficiency['avg_improvement']:.4f}")
    col2.metric("Max Improvement", f"{efficiency['max_improvement']:.4f}")
    col3.metric("Min Improvement", f"{efficiency['min_improvement']:.4f}")
    col4.metric("Total Improvement", f"{efficiency['total_improvement']:.4f}")
    col5.metric("Improvement %", f"{efficiency['improvement_percentage']:.1f}%")

    # Growth rate
    st.subheader("📈 Growth Rate")
    growth_rates = metrics.get_growth_rate(window=10)
    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(x=list(range(len(growth_rates))), y=growth_rates,
                                     name="Growth Rate", line=dict(color="blue", width=2)))
    fig_growth.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_growth.update_layout(title="Generation-over-Generation Growth Rate",
                              xaxis_title="Generation", yaxis_title="Growth Rate", height=300)
    st.plotly_chart(fig_growth, use_container_width=True)

    # Acceleration
    st.subheader("🚀 Acceleration (Rate of Change)")
    acceleration = metrics.get_acceleration(window=10)
    fig_accel = go.Figure()
    fig_accel.add_trace(go.Scatter(x=list(range(len(acceleration))), y=acceleration,
                                    name="Acceleration", line=dict(color="purple", width=2)))
    fig_accel.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_accel.update_layout(title="Acceleration of Fitness Improvement",
                             xaxis_title="Generation", yaxis_title="Acceleration", height=300)
    st.plotly_chart(fig_accel, use_container_width=True)

    # Quantiles
    st.subheader("📊 Quantile Analysis")
    quantiles = metrics.get_quantiles("best")
    q_data = {"Quantile": [f"{q:.0%}" for q in quantiles.keys()],
              "Value": [f"{v:.3f}" for v in quantiles.values()]}
    st.table(q_data)

    # Performance bands
    st.subheader("📏 Performance Bands")
    bands = metrics.get_performance_bands("best", n_bands=5)
    band_data = {"Band": [f"{lo:.0%}-{hi:.0%}" for lo, hi, _ in bands],
                 "Min": [f"{lo:.3f}" for _, _, (lo, _) in bands],
                 "Max": [f"{hi:.3f}" for _, _, (_, hi) in bands]}
    st.table(band_data)

    # Bottleneck detection
    st.subheader("🔍 Bottleneck Detection")
    bottlenecks = metrics.get_bottleneck_generations(threshold=0.001, window=5)
    if bottlenecks:
        st.warning(f"Detected {len(bottlenecks)} bottleneck generations")
        # Show bottleneck regions
        regions = _find_bottleneck_regions(bottlenecks)
        for start, end, length in regions:
            st.code(f"Generations {start}-{end} ({length} gens)")
    else:
        st.success("No significant bottlenecks detected!")

    # Convergence
    st.subheader("🎯 Convergence Analysis")
    convergence = metrics.get_convergence_status(window=20)
    if convergence["converged"]:
        st.success(f"Run appears converged (change={convergence['change']:.6f} in window={convergence['window']})")
    else:
        st.info(f"Not converged (change={convergence['change']:.6f} in window={convergence['window']})")
    st.json(convergence)


def _render_tournament_tab(runs: List, run_manager, tournament_data: Dict, elo_data: Dict, selected_compare: List[str]) -> None:
    """Render tournament visualization tab."""
    st.header("🏆 Tournament Analysis")

    from .tournament_viz import render_tournament_dashboard, render_elo_history_dashboard

    if not selected_compare:
        st.info("Select runs with tournament data in the sidebar.")
        return

    # ELO History
    st.subheader("📈 ELO Rating History")
    for run_id in selected_compare:
        elo = elo_data.get(run_id, {})
        if elo:
            render_elo_history_dashboard(elo, title=f"{run_id} - ELO Progression")
            st.markdown("---")

    # Tournament history
    st.subheader("🏅 Tournament Results")
    for run_id in selected_compare:
        t_history = tournament_data.get(run_id, [])
        if t_history:
            st.markdown(f"### {run_id} - {len(t_history)} Tournaments")
            # Show latest tournament
            latest = t_history[-1] if t_history else {}
            if latest:
                rankings = latest.get("rankings", [])
                if rankings:
                    # Summary metrics
                    elos = [r[1] for r in rankings]
                    st.metric("Top Score", f"{max(elos):.2f}")
                    st.metric("ELO Spread", f"{max(elos) - min(elos):.2f}")

                    # Rankings table
                    ranking_data = {
                        "Rank": [i + 1 for i, _ in enumerate(rankings)],
                        "Agent": [aid for aid, _ in rankings],
                        "Score": [f"{score:.3f}" for _, score in rankings],
                    }
                    st.table(ranking_data)
            st.markdown("---")


def _render_comparison_tab(runs: List, run_manager, fitness_data: Dict, selected_compare: List[str]) -> None:
    """Render run comparison tab."""
    st.header("🔬 Run Comparison")

    if len(selected_compare) < 2:
        st.info("Select at least 2 runs to compare in the sidebar.")
        return

    # Comparison metrics
    st.subheader("📊 Comparison Summary")
    comparison = run_manager.compare_runs(selected_compare, metric="best")

    if comparison.get("summary"):
        summary = comparison["summary"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Overall", f"{summary['best_overall']:.3f}")
        col2.metric("Best Run", summary['best_run'])
        col3.metric("Avg Improvement", f"{summary['avg_improvement']:.4f}")
        col4.metric("Diversity", f"{summary['diversity']:.4f}")

    # Statistical significance
    st.subheader("📐 Statistical Significance")
    if len(selected_compare) >= 2:
        run1_id, run2_id = selected_compare[0], selected_compare[1]
        run1_data = fitness_data.get(run1_id, {}).get("best", [])
        run2_data = fitness_data.get(run2_id, {}).get("best", [])

        if run1_data and run2_data:
            from .metrics import TrainingMetrics
            metrics = TrainingMetrics()
            sig = metrics.compute_statistical_significance(run1_data, run2_data)

            col1, col2, col3 = st.columns(3)
            col1.metric("t-statistic", f"{sig['t_statistic']:.3f}")
            col2.metric("p-value", f"{sig['p_value']:.6f}")
            col3.metric("Significant", "Yes" if sig['significant'] else "No")

            if sig['significant']:
                st.success(f"Runs are statistically different (p={sig['p_value']:.6f} < 0.05)")
            else:
                st.info(f"No significant difference (p={sig['p_value']:.6f} >= 0.05)")

    # Side-by-side fitness curves
    st.subheader("📈 Fitness Curves Comparison")
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, run_id in enumerate(selected_compare):
        data = fitness_data.get(run_id, {})
        values = data.get("best", [])
        if values:
            generations = list(range(1, len(values) + 1))
            fig.add_trace(go.Scatter(x=generations, y=values, name=run_id,
                                     line=dict(color=colors[i % len(colors)], width=2)))

    fig.update_layout(height=400, showlegend=True, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def _render_runs_tab(runs: List, run_manager, fitness_data: Dict, tournament_data: Dict) -> None:
    """Render runs overview tab."""
    st.header("📦 All Runs")

    if not runs:
        st.info("No runs discovered.")
        return

    # Runs table
    runs_data = {
        "Run ID": [r.run_id for r in runs],
        "Name": [r.name for r in runs],
        "Best Fitness": [f"{r.best_fitness:.3f}" for r in runs],
        "Generations": [f"{r.actual_generations}/{r.max_generations}" for r in runs],
        "Progress": [f"{r.progress:.1%}" for r in runs],
        "Status": ["Complete" if r.is_complete else "Running" for r in runs],
    }
    st.table(runs_data)

    # Tags
    tags = run_manager.get_run_tags()
    if tags:
        st.subheader("🏷️ Tags")
        for tag, run_ids in tags.items():
            st.text(f"{tag}: {', '.join(run_ids)}")


def _render_config_tab(runs: List, run_manager) -> None:
    """Render configuration comparison tab."""
    st.header("⚙️ Configuration Comparison")

    if not runs:
        st.info("No runs to compare.")
        return

    # Show configs for selected runs
    st.subheader("Active Run Configuration")
    if runs:
        config = runs[0].config
        if config:
            st.json(config)
        else:
            st.info("No configuration data available.")


# =============================================================================
# Card Meta Analysis Tab
# =============================================================================


def _render_card_meta_tab(runs: List, run_manager, fitness_data: Dict) -> None:
    """Render card usage and meta analysis tab.
    
    Analyzes which cards are most effective at different stages of training,
    showing win rates by card type, elixir cost distribution, and more.
    """
    st.header("🃏 Card Meta Analysis")
    st.markdown("Analysis of card usage patterns, effectiveness, and meta trends from simulation data.")

    # Card registry info
    try:
        from src.env.sim.entities import CARD_DEFS
        st.subheader("📚 Card Registry Overview")
        total_cards = len(CARD_DEFS)
        
        # Count by cost
        by_cost = {}
        by_type = {}
        for name, card in CARD_DEFS.items():
            cost_key = f"{int(card.cost)}elixir" if card.cost == int(card.cost) else f"{card.cost}elixir"
            by_cost[cost_key] = by_cost.get(cost_key, 0) + 1
            by_type[card.card_type] = by_type.get(card.card_type, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cards", total_cards)
        with col2:
            st.metric("Unit Types", len(by_type))
        with col3:
            costs = sorted(by_cost.keys(), key=lambda x: float(x.replace('elixir', '')))
            st.metric("Elixir Range", f"{costs[0]} - {costs[-1]}") if costs else st.metric("Elixir Range", "N/A")
        
        # Card type breakdown
        st.markdown("### Cards by Type")
        type_data = {"Type": list(by_type.keys()), "Count": list(by_type.values())}
        st.bar_chart(type_data.set_index("Type"))
        
        # Elixir cost distribution
        st.markdown("### Cards by Elixir Cost")
        cost_data = {"Elixir Cost": list(by_cost.keys()), "Count": list(by_cost.values())}
        st.bar_chart(cost_data.set_index("Elixir Cost"))
        
    except Exception as e:
        st.warning(f"Could not load card registry: {e}")

    # Training data analysis
    if runs:
        st.markdown("---")
        st.subheader("📊 Training Data Analysis")
        
        # Show available run data
        selected_run = st.selectbox("Select Run for Analysis", [r.run_id for r in runs])
        
        if selected_run:
            fitness = fitness_data.get(selected_run, {})
            if fitness:
                best_fitness = fitness.get("best", [])
                mean_fitness = fitness.get("mean", [])
                diversity = fitness.get("diversity", [])
                std_fitness = fitness.get("std", [])
                
                col1, col2, col3, col4 = st.columns(4)
                if best_fitness:
                    col1.metric("Final Best Fitness", f"{best_fitness[-1]:.4f}")
                    col2.metric("Best Improvement", f"{max(best_fitness) - min(best_fitness):.4f}" if len(best_fitness) > 1 else "N/A")
                if mean_fitness:
                    col3.metric("Final Mean Fitness", f"{mean_fitness[-1]:.4f}")
                if diversity:
                    col4.metric("Final Diversity", f"{diversity[-1]:.4f}")
                
                # Fitness evolution chart
                if best_fitness and mean_fitness:
                    st.markdown("### Fitness Evolution")
                    import plotly.graph_objects as go
                    from plotly.subplots import make_subplots
                    
                    gens = list(range(1, len(best_fitness) + 1))
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Best Fitness", "Mean Fitness"),
                        vertical_spacing=0.1,
                    )
                    
                    fig.add_trace(go.Scatter(x=gens, y=best_fitness, name="Best", line=dict(color="green", width=2)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=gens, y=mean_fitness, name="Mean", line=dict(color="blue", width=1.5)), row=2, col=1)
                    
                    # Add shaded std region
                    if std_fitness:
                        upper = [b + s for b, s in zip(best_fitness, std_fitness)]
                        lower = [max(0, b - s) for b, s in zip(best_fitness, std_fitness)]
                        fig.add_trace(go.Scatter(x=gens, y=upper, name="±1σ", fill=None, showlegend=False,
                                                 line=dict(width=0)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=gens, y=lower, name="±1σ", fill='tonexty', fillcolor='rgba(0,0,255,0.1)',
                                                 showlegend=False, line=dict(width=0)), row=1, col=1)
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Convergence analysis
                if len(best_fitness) > 20:
                    st.markdown("### Convergence Analysis")
                    from src.viz.metrics import TrainingMetrics
                    metrics = TrainingMetrics()
                    metrics.history = {"best": best_fitness, "mean": mean_fitness}
                    
                    conv = metrics.get_convergence_status(window=20)
                    growth = metrics.get_growth_rate(window=10)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        status = "✅ Converged" if conv["converged"] else "🔄 Not Converged"
                        st.metric("Convergence Status", status)
                    with col2:
                        recent_growth = np.mean(growth[-5:]) if growth else 0
                        st.metric("Recent Growth Rate", f"{recent_growth:.6f}")
                    with col3:
                        gens_to_half = metrics.get_generations_to_half_loss(best_fitness)
                        st.metric("Gens to Half Loss", str(gens_to_half) if gens_to_half else "N/A")
    
    # Card type effectiveness guide
    st.markdown("---")
    st.subheader("📖 Card Type Effectiveness Guide")
    
    guide_data = {
        "Card Type": [
            "Tank", "Melee", "Ranged", "Air", "Swarm", "Building",
            "Spell (Damage)", "Spell (Control)", "Spell (Utility)"
        ],
        "Typical Cost Range": [
            "5-8", "2-4", "3-5", "3-7", "1-5", "2-6",
            "3-6", "2-4", "3-5"
        ],
        "Win Rate Impact": [
            "High (late game)", "Medium", "High (consistent)", "High (air control)",
            "Situational", "Variable",
            "High (direct damage)", "Medium (control)", "Medium-High (utility)"
        ],
        "Best Matchup": [
            "Ground units", "Swarm/Tanks", "Air/Ground", "Ground buildings",
            "Towers/Buildings", "Any (defense)",
            "Swarm/Horde", "Single targets", "Control situations"
        ],
    }
    st.dataframe(guide_data, use_container_width=True)


# =============================================================================
# Helper Functions
# =============================================================================


def _moving_average(values: List[float], window: int) -> List[float]:
    """Compute moving average."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start:i + 1])))
    return result


def _exponential_moving_average(values: List[float], alpha: float) -> List[float]:
    """Compute exponential moving average."""
    if not values:
        return []
    ema = [values[0]]
    for i in range(1, len(values)):
        ema.append(alpha * values[i] + (1 - alpha) * ema[-1])
    return ema


def _savitzky_golay(values: List[float], window: int) -> List[float]:
    """Simple Savitzky-Golay filter approximation."""
    if len(values) < window:
        return values
    half = window // 2
    result = list(values)
    for i in range(half, len(values) - half):
        result[i] = float(np.mean(values[i-half:i+half+1]))
    return result


def _find_bottleneck_regions(bottlenecks: List[int], gap_threshold: int = 5) -> List[tuple]:
    """Find contiguous regions of bottlenecks."""
    if not bottlenecks:
        return []

    regions = []
    start = bottlenecks[0]
    prev = bottlenecks[0]

    for b in bottlenecks[1:]:
        if b - prev <= gap_threshold:
            prev = b
        else:
            regions.append((start, prev, prev - start + 1))
            start = b
            prev = b

    regions.append((start, prev, prev - start + 1))
    return regions


# Keep the original function for backwards compatibility
def run_dashboard(runs_dir: str = "runs", refresh_interval: int = 5) -> None:
    """Run the training visualization dashboard (legacy wrapper)."""
    run_advanced_dashboard(runs_dir, refresh_interval)


def _render_monitoring_tab(runs: List, run_manager) -> None:
    """Render the monitoring/resource utilization tab."""
    st.header("📡 Resource Monitoring & Performance")

    # Load monitoring data
    import os
    monitoring_dir = Path("runs/monitoring")
    snapshots = []
    if monitoring_dir.exists():
        for f in sorted(monitoring_dir.glob("snapshot_gen_*.json")):
            try:
                with open(f) as fh:
                    snapshots.append(json.load(fh))
            except:
                pass

    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_gens = len(snapshots)
        st.metric("Total Generations Monitored", total_gens)
    with col2:
        if snapshots:
            latest = snapshots[-1]
            speed = latest.get("speed", {})
            st.metric("Gen/Hour", f"{speed.get('generations_per_hour', 0):.1f}")
        else:
            st.metric("Gen/Hour", "N/A")
    with col3:
        if snapshots:
            latest = snapshots[-1]
            conv = latest.get("convergence", {})
            trend = conv.get("trend", "N/A")
            st.metric("Convergence Trend", trend)
        else:
            st.metric("Convergence Trend", "N/A")
    with col4:
        # Check for alert history
        alert_path = Path("runs/alert_history.json")
        if alert_path.exists():
            try:
                with open(alert_path) as f:
                    alerts = json.load(f)
                alert_count = len(alerts) if isinstance(alerts, list) else 0
            except:
                alert_count = 0
        else:
            alert_count = 0
        st.metric("Alerts Generated", alert_count)

    st.markdown("---")

    # Fitness with monitoring overlay
    if snapshots:
        st.subheader("📈 Fitness with Resource Correlation")
        latest = snapshots[-1]
        best = latest.get("best_fitness", [])
        mean = latest.get("mean_fitness", [])
        diversity = latest.get("diversity", [])
        gens = latest.get("generation_labels", [])

        if best and gens:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Best Fitness", "Mean Fitness", "Population Diversity"),
                vertical_spacing=0.08,
                row_heights=[0.4, 0.3, 0.3],
            )

            fig.add_trace(go.Scatter(x=gens, y=best, name="Best", line=dict(color="green", width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=gens, y=mean, name="Mean", line=dict(color="blue", width=1.5)), row=2, col=1)
            fig.add_trace(go.Scatter(x=gens, y=diversity, name="Diversity", line=dict(color="orange", width=1.5)), row=3, col=1)

            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # Resource utilization summary
    st.subheader("🖥️ Resource Utilization Summary")
    resource_summary = {}
    for snap in snapshots[-10:]:
        if "gpu" in snap:
            resource_summary = snap["gpu"]

    if resource_summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg GPU Memory", f"{resource_summary.get('avg_memory_gb', 0):.1f} GB")
        with col2:
            st.metric("Max GPU Memory", f"{resource_summary.get('max_memory_gb', 0):.1f} GB")
        with col3:
            st.metric("Avg Compute", f"{resource_summary.get('avg_compute', 0):.1f}%")
        with col4:
            st.metric("Avg Temperature", f"{resource_summary.get('avg_temperature', 0):.1f}°C")
    else:
        st.info("No GPU monitoring data available. Enable `monitor_resources: true` in config.")

    # Bottleneck history
    st.subheader("🔍 Bottleneck Detection History")
    if snapshots:
        bottleneck_counts = {}
        for snap in snapshots[-50:]:
            convergence = snap.get("convergence", {})
            trend = convergence.get("trend", "unknown")
            bottleneck_counts[trend] = bottleneck_counts.get(trend, 0) + 1

        if bottleneck_counts:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Pie(
                labels=list(bottleneck_counts.keys()),
                values=list(bottleneck_counts.values()),
                hole=0.3,
            )])
            fig.update_layout(title="Training Trend Distribution (last 50 gens)", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Alert history
    st.subheader("🔔 Alert History")
    alert_path = Path("runs/alert_history.json")
    if alert_path.exists():
        try:
            with open(alert_path) as f:
                alerts = json.load(f)
            if isinstance(alerts, list) and alerts:
                alert_df = []
                for a in alerts[-20:]:
                    alert_df.append({
                        "Type": a.get("alert_type", "N/A"),
                        "Level": a.get("level", "N/A"),
                        "Message": a.get("message", "N/A"),
                        "Time": a.get("timestamp", 0),
                    })
                st.dataframe(alert_df, use_container_width=True)
            else:
                st.info("No alerts generated during training.")
        except:
            st.info("Could not load alert history.")
    else:
        st.info("No alert history found. Enable `enable_alerts: true` in config.")

    # Model registry
    st.subheader("📦 Model Registry")
    registry_path = Path("runs/model_registry/registry.json")
    if registry_path.exists():
        try:
            with open(registry_path) as f:
                reg_data = json.load(f)
            models = reg_data.get("models", [])
            if models:
                model_df = []
                for m in models[-10:]:
                    model_df.append({
                        "ID": m.get("model_id", "N/A"),
                        "Version": m.get("version", "N/A"),
                        "Fitness": m.get("fitness", 0),
                        "Stage": m.get("stage", "N/A"),
                        "Architecture": m.get("architecture", "N/A"),
                    })
                st.dataframe(model_df, use_container_width=True)
                st.info(f"Showing latest 10 of {len(models)} registered models")
            else:
                st.info("No models registered yet. Enable `enable_registry: true` in config.")
        except:
            st.info("Could not load registry.")
    else:
        st.info("No model registry found. Enable `enable_registry: true` in config.")

    # Configuration hints
    st.markdown("---")
    st.subheader("⚙️ Enable Monitoring & Alerting")
    st.markdown("""
To enable these features, add to your evolution.yaml config:

```yaml
monitoring:
  enabled: true
  sample_interval: 1.0

alerting:
  enabled: true
  log_path: runs/alerts.log

registry:
  enabled: true
  dir: runs/model_registry
```

Or via CLI:
```bash
crp train --monitor-resources --enable-alerts --enable-registry
```""")


def render_fitness_curves(fitness_history: Dict[str, List[float]],
                          title: str = "Fitness Over Generations") -> None:
    """Render fitness curves using Plotly (legacy wrapper)."""
    if not HAS_STREAMLIT:
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    generations = list(range(1, len(fitness_history.get("best", [1])) + 1))

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Best Fitness", "Population Statistics", "Diversity"),
        vertical_spacing=0.12,
    )

    if fitness_history.get("best"):
        fig.add_trace(go.Scatter(x=generations, y=fitness_history["best"],
                                  name="Best", line=dict(color="green", width=2)), row=1, col=1)

    colors = {"mean": "blue", "median": "orange", "min": "red", "max": "purple"}
    for key, color in colors.items():
        if fitness_history.get(key):
            fig.add_trace(go.Scatter(x=generations, y=fitness_history[key],
                                      name=key.capitalize(), line=dict(color=color, width=1.5)), row=2, col=1)

    if fitness_history.get("diversity"):
        fig.add_trace(go.Scatter(x=generations, y=fitness_history["diversity"],
                                  name="Diversity", line=dict(color="cyan", width=1.5)), row=3, col=1)

    fig.update_layout(height=700, title_text=title, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
