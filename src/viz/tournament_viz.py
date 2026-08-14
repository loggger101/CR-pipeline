"""Tournament visualization utilities.

Provides:
- Bracket visualization (ASCII and Plotly)
- Head-to-head record charts
- ELO rating progression charts
- Tournament summary statistics
- Integration with Streamlit dashboard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..train import TournamentResult, TournamentBracket

logger = logging.getLogger(__name__)

# Dashboard is optional - only import if streamlit is available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# ASCII Bracket Renderer
# =============================================================================


def render_bracket_ascii(bracket: "TournamentBracket", max_width: int = 80) -> str:
    """Render a tournament bracket as ASCII art.

    Args:
        bracket: TournamentBracket to render.
        max_width: Maximum width of the output.

    Returns:
        ASCII art string representation.
    """
    if bracket is None:
        return "No bracket data available."

    lines = []
    lines.append(f"{'=' * max_width}")
    lines.append(f"Tournament Bracket: {bracket.format.name}")
    lines.append(f"{'=' * max_width}")

    for round_num, round_matches in enumerate(bracket.rounds, 1):
        lines.append(f"\n--- Round {round_num} ---")
        for agent1, agent2 in round_matches:
            if agent2 is None:
                lines.append(f"  {agent1 or 'Unknown'}  [BYE]")
            else:
                lines.append(f"  {agent1 or 'Unknown':<15} vs {agent2 or 'Unknown':<15}")

    if bracket.winners:
        lines.append(f"\n--- Winners ---")
        for winner in bracket.winners:
            lines.append(f"  >>> {winner}")

    if bracket.loser_bracket:
        lines.append(f"\n--- Loser Bracket ---")
        for agent in bracket.loser_bracket:
            lines.append(f"  {agent}")

    lines.append(f"\n{'=' * max_width}")
    return "\n".join(lines)


# =============================================================================
# ELO Progression Chart
# =============================================================================


def create_elo_progression_chart(
    elo_history: Dict[str, List[float]],
    title: str = "ELO Rating Progression",
    height: int = 500,
) -> Optional[go.Figure]:
    """Create a Plotly chart showing ELO rating progression over generations.

    Args:
        elo_history: Dict mapping agent_id to list of ELO ratings per generation.
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        Plotly Figure object, or None if Plotly unavailable.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available. Install with: pip install plotly")
        return None

    if not elo_history:
        return None

    fig = go.Figure()

    # Sort agents by final ELO for consistent coloring
    sorted_agents = sorted(elo_history.keys(),
                           key=lambda aid: elo_history[aid][-1] if elo_history[aid] else 0,
                           reverse=True)

    # Limit to top agents to avoid clutter
    display_agents = sorted_agents[:10] if len(sorted_agents) > 10 else sorted_agents

    for agent_id in display_agents:
        ratings = elo_history[agent_id]
        if not ratings:
            continue
        generations = list(range(1, len(ratings) + 1))
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=ratings,
                mode="lines+markers",
                name=agent_id,
                line=dict(width=2),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Generation",
        yaxis_title="ELO Rating",
        yaxis=dict(range=[min(min(r) for r in elo_history.values()) - 50,
                          max(max(r) for r in elo_history.values()) + 50]),
        height=height,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
    )

    return fig


# =============================================================================
# Win Rate Chart
# =============================================================================


def create_win_rate_chart(
    rankings: List[Tuple[str, float]],
    agent_stats: Dict[str, any],
    title: str = "Agent Win Rates",
    height: int = 400,
) -> Optional[go.Figure]:
    """Create a bar chart showing agent win rates.

    Args:
        rankings: List of (agent_id, score) tuples from tournament.
        agent_stats: Dict of agent_id -> AgentTournamentStats.
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        Plotly Figure object, or None if Plotly unavailable.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available.")
        return None

    agents = [aid for aid, _ in rankings]
    win_rates = [agent_stats[aid].win_rate if aid in agent_stats else 0.0 for aid in agents]
    losses = [agent_stats[aid].losses if aid in agent_stats else 0 for aid in agents]
    draws = [agent_stats[aid].draws if aid in agent_stats else 0 for aid in agents]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=agents,
        y=win_rates,
        name="Win Rate",
        marker_color="green",
        text=[f"{wr:.1%}" for wr in win_rates],
        textposition="auto",
    ))

    # Stack losses and draws below
    fig.add_trace(go.Bar(
        x=agents,
        y=[losses[i] / max(1, agent_stats[aid].matches) for i, aid in enumerate(agents)],
        name="Loss Rate",
        marker_color="red",
    ))

    fig.add_trace(go.Bar(
        x=agents,
        y=[draws[i] / max(1, agent_stats[aid].matches) for i, aid in enumerate(agents)],
        name="Draw Rate",
        marker_color="orange",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Agent",
        yaxis_title="Rate",
        height=height,
        barmode="stack",
        yaxis=dict(range=[0, 1.05]),
        showlegend=True,
    )

    return fig


# =============================================================================
# Head-to-Head Matrix Chart
# =============================================================================


def create_h2h_matrix_chart(
    h2h_records: Dict[Tuple[str, str], any],
    agent_ids: List[str],
    title: str = "Head-to-Head Win Rates",
    height: int = 600,
) -> Optional[go.Figure]:
    """Create a heatmap showing head-to-head win rates.

    Args:
        h2h_records: Dict of (agent1, agent2) -> HeadToHeadRecord.
        agent_ids: List of all agent IDs.
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        Plotly Figure object, or None if Plotly unavailable.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available.")
        return None

    n = len(agent_ids)
    matrix = np.zeros((n, n))

    for i, aid1 in enumerate(agent_ids):
        for j, aid2 in enumerate(agent_ids):
            if i == j:
                matrix[i, j] = 0.5  # Self-play = draw
            else:
                key = (min(aid1, aid2), max(aid1, aid2))
                record = h2h_records.get(key)
                if record and record.matches > 0:
                    # Win rate for row agent (agent1 perspective)
                    matrix[i, j] = record.agent1_wins / record.matches if aid1 == record.agent1_id else record.agent2_wins / record.matches
                else:
                    matrix[i, j] = 0.5  # No data = assumed draw

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=agent_ids,
        y=agent_ids,
        colorscale="RdYlGn",
        text=[[f"{matrix[i, j]:.2%}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        hoverongaps=False,
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Agent (column wins)",
        yaxis_title="Agent (row wins)",
        xaxis=dict(side="bottom"),
    )

    return fig


# =============================================================================
# Tournament Summary Statistics
# =============================================================================


@dataclass
class TournamentSummary:
    """Structured tournament summary for visualization.

    Attributes:
        total_agents: Number of agents in tournament.
        total_matches: Total matches played.
        avg_win_rate: Average win rate across agents.
        elo_spread: Difference between highest and lowest ELO.
        top_agent: ID of top-ranked agent.
        top_elo: ELO of top agent.
        competitiveness: Measure of how close the tournament was (1.0 = very competitive).
    """
    total_agents: int = 0
    total_matches: int = 0
    avg_win_rate: float = 0.0
    elo_spread: float = 0.0
    top_agent: str = ""
    top_elo: float = 0.0
    competitiveness: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_agents": self.total_agents,
            "total_matches": self.total_matches,
            "avg_win_rate": self.avg_win_rate,
            "elo_spread": self.elo_spread,
            "top_agent": self.top_agent,
            "top_elo": self.top_elo,
            "competitiveness": self.competitiveness,
        }


def compute_tournament_summary(result: "TournamentResult") -> TournamentSummary:
    """Compute summary statistics from a tournament result.

    Args:
        result: TournamentResult to summarize.

    Returns:
        TournamentSummary with computed statistics.
    """
    n = len(result.rankings)
    total_matches = result.total_matches

    # Average win rate
    win_rates = [result.get_win_rate(aid) for aid, _ in result.rankings]
    avg_win_rate = np.mean(win_rates) if win_rates else 0.0

    # ELO spread
    elos = list(result.elo_ratings.values())
    elo_spread = max(elos) - min(elos) if elos else 0.0

    # Top agent
    top_agent, top_score = result.rankings[0] if result.rankings else ("", 0.0)
    top_elo = result.elo_ratings.get(top_agent, 0.0)

    # Competitiveness: lower Elo spread = more competitive
    # Normalize: 1.0 = perfectly competitive (all equal), 0.0 = not competitive
    max_elo = max(elos) if elos else 1500.0
    competitiveness = max(0.0, 1.0 - elo_spread / max_elo) if max_elo > 0 else 0.0

    return TournamentSummary(
        total_agents=n,
        total_matches=total_matches,
        avg_win_rate=float(avg_win_rate),
        elo_spread=float(elo_spread),
        top_agent=top_agent,
        top_elo=float(top_elo),
        competitiveness=float(competitiveness),
    )


# =============================================================================
# Streamlit Dashboard Integration
# =============================================================================


def render_tournament_dashboard(result: "TournamentResult", generation: int = 0) -> None:
    """Render tournament results in Streamlit dashboard.

    Args:
        result: TournamentResult to display.
        generation: Generation number.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader(f"Tournament Results - Generation {generation}")

    # Summary statistics
    summary = compute_tournament_summary(result)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agents", summary.total_agents)
    col2.metric("Matches", summary.total_matches)
    col3.metric("Avg Win Rate", f"{summary.avg_win_rate:.1%}")
    col4.metric("ELO Spread", f"{summary.elo_spread:.0f}")

    # Competitiveness indicator
    comp_level = "Very Competitive" if summary.competitiveness > 0.8 else \
                 "Competitive" if summary.competitiveness > 0.5 else \
                 "Lopsided" if summary.competitiveness > 0.2 else "One-sided"
    st.info(f"Tournament Competitiveness: {comp_level}")

    # Rankings table
    st.markdown("### Rankings")
    ranking_data = {
        "Rank": [i + 1 for i, _ in result.rankings],
        "Agent": [aid for aid, _ in result.rankings],
        "Score": [f"{score:.3f}" for _, score in result.rankings],
        "ELO": [result.elo_ratings.get(aid, 0) for aid, _ in result.rankings],
    }
    st.table(ranking_data)

    # Charts
    if HAS_PLOTLY:
        col1, col2 = st.columns(2)
        with col1:
            win_rate_fig = create_win_rate_chart(
                result.rankings, result.agent_stats
            )
            if win_rate_fig:
                st.plotly_chart(win_rate_fig, use_container_width=True)

        with col2:
            elo_fig = create_elo_progression_chart(result.elo_ratings)
            if elo_fig:
                st.plotly_chart(elo_fig, use_container_width=True)

        # H2H matrix if enough data
        if result.h2h_records:
            agent_ids = [aid for aid, _ in result.rankings]
            h2h_fig = create_h2h_matrix_chart(result.h2h_records, agent_ids)
            if h2h_fig:
                st.markdown("### Head-to-Head Matrix")
                st.plotly_chart(h2h_fig, use_container_width=True)

    # Bracket visualization
    if result.bracket:
        st.markdown("### Bracket")
        st.code(render_bracket_ascii(result.bracket), language="text")

    # H2H Records
    if result.h2h_records:
        st.markdown("### Head-to-Head Records")
        h2h_data = {}
        for (a1, a2), record in result.h2h_records.items():
            h2h_data[f"{a1} vs {a2}"] = (
                f"{record.agent1_wins}W/{record.agent2_wins}L/{record.draws}D"
            )
        for k, v in h2h_data.items():
            st.text(f"{k}: {v}")


def render_elo_history_dashboard(elo_history: Dict[str, List[float]],
                                  title: str = "ELO Rating History") -> None:
    """Render ELO history chart in Streamlit.

    Args:
        elo_history: Dict of agent_id -> ELO history.
        title: Chart title.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.subheader(title)

    if HAS_PLOTLY and elo_history:
        fig = create_elo_progression_chart(elo_history, title)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    elif not elo_history:
        st.info("No ELO history data available.")


# =============================================================================
# JSON Serialization Helpers
# =============================================================================


def tournament_result_to_dict(result: "TournamentResult") -> dict:
    """Convert TournamentResult to JSON-serializable dictionary.

    Args:
        result: TournamentResult to convert.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        "generation": result.generation,
        "total_matches": result.total_matches,
        "rankings": [(aid, float(score)) for aid, score in result.rankings],
        "elo_ratings": {aid: float(elo) for aid, elo in result.elo_ratings.items()},
        "agent_stats": {
            aid: {
                "wins": stats.wins,
                "draws": stats.draws,
                "losses": stats.losses,
                "towers_destroyed": stats.towers_destroyed,
                "total_duration": stats.total_duration,
                "elo_rating": stats.elo_rating,
                "win_rate": stats.win_rate,
            }
            for aid, stats in result.agent_stats.items()
        },
        "h2h_records": {
            f"{a1}|{a2}": {
                "agent1_wins": rec.agent1_wins,
                "agent2_wins": rec.agent2_wins,
                "draws": rec.draws,
                "agent1_towers": rec.agent1_towers,
                "agent2_towers": rec.agent2_towers,
                "agent1_duration": rec.agent1_duration,
                "agent2_duration": rec.agent2_duration,
            }
            for (a1, a2), rec in result.h2h_records.items()
        },
        "summary": compute_tournament_summary(result).to_dict(),
    }


def tournament_result_from_dict(data: dict) -> "TournamentResult":
    """Reconstruct TournamentResult from dictionary.

    Args:
        data: Dictionary from JSON deserialization.

    Returns:
        Reconstructed TournamentResult.
    """
    from ..train import TournamentResult, AgentTournamentStats, HeadToHeadRecord

    # Reconstruct agent stats
    agent_stats = {}
    for aid, stats_data in data.get("agent_stats", {}).items():
        agent_stats[aid] = AgentTournamentStats(
            agent_id=aid,
            wins=stats_data.get("wins", 0),
            draws=stats_data.get("draws", 0),
            losses=stats_data.get("losses", 0),
            towers_destroyed=stats_data.get("towers_destroyed", 0),
            total_duration=stats_data.get("total_duration", 0),
            elo_rating=stats_data.get("elo_rating", 1500.0),
        )

    # Reconstruct H2H records
    h2h_records = {}
    for key, rec_data in data.get("h2h_records", {}).items():
        a1, a2 = key.split("|")
        h2h_records[(a1, a2)] = HeadToHeadRecord(
            agent1_id=a1,
            agent2_id=a2,
            agent1_wins=rec_data.get("agent1_wins", 0),
            agent2_wins=rec_data.get("agent2_wins", 0),
            draws=rec_data.get("draws", 0),
            agent1_towers=rec_data.get("agent1_towers", 0),
            agent2_towers=rec_data.get("agent2_towers", 0),
            agent1_duration=rec_data.get("agent1_duration", 0),
            agent2_duration=rec_data.get("agent2_duration", 0),
        )

    return TournamentResult(
        rankings=data.get("rankings", []),
        agent_stats=agent_stats,
        h2h_records=h2h_records,
        wins={aid: stats.wins for aid, stats in agent_stats.items()},
        draws={aid: stats.draws for aid, stats in agent_stats.items()},
        losses={aid: stats.losses for aid, stats in agent_stats.items()},
        total_matches=data.get("total_matches", 0),
        elo_ratings=data.get("elo_ratings", {}),
        generation=data.get("generation", 0),
    )
