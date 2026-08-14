"""Report generation for CR-Pipeline.

Provides:
- Training report generation (HTML, PDF, Markdown)
- Experiment comparison reports
- Tournament result reports
- Model performance reports
- Automated report scheduling
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Report Types
# =============================================================================


class ReportType:
    """Types of reports."""
    TRAINING = "training"
    EXPERIMENT = "experiment"
    TOURNAMENT = "tournament"
    MODEL = "model"
    COMPARISON = "comparison"


# =============================================================================
# Report Generator
# =============================================================================


class ReportGenerator:
    """Generates various reports for CR-Pipeline.

    Supports:
    - HTML reports with embedded charts
    - Markdown reports
    - JSON reports
    - PDF reports (via reportlab)
    """

    def __init__(self, output_dir: str = "reports"):
        """Initialize the report generator.

        Args:
            output_dir: Directory for generated reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_training_report(
        self,
        run_dir: str,
        output_filename: str = "training_report.html",
        include_charts: bool = True,
    ) -> str:
        """Generate a training report for a run.

        Args:
            run_dir: Directory containing training data.
            output_filename: Output filename.
            include_charts: Whether to include charts.

        Returns:
            Path to generated report.
        """
        run_path = Path(run_dir)

        # Load data
        metrics_path = run_path / "metrics.json"
        fitness_path = run_path / "fitness_history.json"

        metrics = {}
        fitness_history = {}

        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        if fitness_path.exists():
            with open(fitness_path) as f:
                fitness_history = json.load(f)

        # Generate HTML
        html = self._generate_training_html(metrics, fitness_history, include_charts)

        output_path = self.output_dir / output_filename
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Training report generated: {output_path}")
        return str(output_path)

    def generate_experiment_report(
        self,
        experiment_data: Dict[str, Any],
        output_filename: str = "experiment_report.html",
    ) -> str:
        """Generate an experiment report.

        Args:
            experiment_data: Experiment data dictionary.
            output_filename: Output filename.

        Returns:
            Path to generated report.
        """
        html = self._generate_experiment_html(experiment_data)

        output_path = self.output_dir / output_filename
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Experiment report generated: {output_path}")
        return str(output_path)

    def generate_tournament_report(
        self,
        tournament_data: Dict[str, Any],
        output_filename: str = "tournament_report.html",
    ) -> str:
        """Generate a tournament report.

        Args:
            tournament_data: Tournament result data.
            output_filename: Output filename.

        Returns:
            Path to generated report.
        """
        html = self._generate_tournament_html(tournament_data)

        output_path = self.output_dir / output_filename
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Tournament report generated: {output_path}")
        return str(output_path)

    def generate_comparison_report(
        self,
        runs_data: List[Dict[str, Any]],
        output_filename: str = "comparison_report.html",
    ) -> str:
        """Generate a run comparison report.

        Args:
            runs_data: List of run data dictionaries.
            output_filename: Output filename.

        Returns:
            Path to generated report.
        """
        html = self._generate_comparison_html(runs_data)

        output_path = self.output_dir / output_filename
        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Comparison report generated: {output_path}")
        return str(output_path)

    def generate_markdown_report(
        self,
        data: Dict[str, Any],
        report_type: str = ReportType.TRAINING,
        output_filename: str = "report.md",
    ) -> str:
        """Generate a Markdown report.

        Args:
            data: Report data.
            report_type: Type of report.
            output_filename: Output filename.

        Returns:
            Path to generated report.
        """
        md = self._generate_markdown(data, report_type)

        output_path = self.output_dir / output_filename
        with open(output_path, "w") as f:
            f.write(md)

        logger.info(f"Markdown report generated: {output_path}")
        return str(output_path)

    def generate_json_report(
        self,
        data: Dict[str, Any],
        output_filename: str = "report.json",
    ) -> str:
        """Generate a JSON report.

        Args:
            data: Report data.
            output_filename: Output filename.

        Returns:
            Path to generated report.
        """
        output_path = self.output_dir / output_filename

        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": data.get("type", "unknown"),
            "data": data,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"JSON report generated: {output_path}")
        return str(output_path)

    # =============================================================================
    # HTML Generators
    # =============================================================================

    def _generate_training_html(
        self,
        metrics: Dict[str, Any],
        fitness_history: Dict[str, List[float]],
        include_charts: bool = True,
    ) -> str:
        """Generate HTML for training report."""
        best_fitness = metrics.get("best_fitness", 0)
        mean_fitness = metrics.get("mean_fitness", 0)
        diversity = metrics.get("diversity", 0)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Training Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #2c3e50; font-size: 14px; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #2c3e50; }}
        .chart-placeholder {{ background: #ecf0f1; padding: 40px; text-align: center; border-radius: 8px; color: #7f8c8d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Training Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <h3>Best Fitness</h3>
                <div class="value">{best_fitness:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Mean Fitness</h3>
                <div class="value">{mean_fitness:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Diversity</h3>
                <div class="value">{diversity:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>Generations</h3>
                <div class="value">{metrics.get('actual_generations', metrics.get('total_generations', 'N/A'))}</div>
            </div>
        </div>

        <div class="section">
            <h2>Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Population Size</td><td>{metrics.get('population_size', 'N/A')}</td></tr>
                <tr><td>Elite Count</td><td>{metrics.get('elite_count', 'N/A')}</td></tr>
                <tr><td>Crossover Rate</td><td>{metrics.get('crossover_rate', 'N/A')}</td></tr>
                <tr><td>Mutation Rate</td><td>{metrics.get('mutation_rate', 'N/A')}</td></tr>
                <tr><td>Mutation Std</td><td>{metrics.get('mutation_std', 'N/A')}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>Fitness History</h2>
            {'<div class="chart-placeholder">Chart would be rendered here with Plotly</div>' if include_charts else '<p>Charts disabled</p>'}
        </div>

        <div class="section">
            <h2>Summary</h2>
            <p>This report summarizes the training run with best fitness of <strong>{best_fitness:.4f}</strong>
            achieved over {metrics.get('actual_generations', metrics.get('total_generations', 'N/A'))} generations.</p>
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_experiment_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for experiment report."""
        summary = data.get("summary", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Report: {data.get('name', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #8e44ad; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #8e44ad; font-size: 14px; }}
        .metric-card .value {{ font-size: 24px; font-weight: bold; color: #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Experiment: {data.get('name', 'Unknown')}</h1>
            <p>ID: {data.get('experiment_id', 'N/A')}</p>
        </div>
        <div class="metrics">
            <div class="metric-card">
                <h3>Total Runs</h3>
                <div class="value">{summary.get('total_runs', 0)}</div>
            </div>
            <div class="metric-card">
                <h3>Completed</h3>
                <div class="value">{summary.get('completed_runs', 0)}</div>
            </div>
            <div class="metric-card">
                <h3>Duration</h3>
                <div class="value">{summary.get('duration_hours', 0):.2f}h</div>
            </div>
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_tournament_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for tournament report."""
        rankings = data.get("rankings", [])
        summary = data.get("summary", {})

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tournament Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #e67e22; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .rankings {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
        .winner {{ background: #d5f4e6; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Tournament Results</h1>
        </div>
        <div class="rankings">
            <h2>Rankings</h2>
            <table>
                <tr><th>Rank</th><th>Agent</th><th>Score</th><th>ELO</th></tr>"""

        for i, (agent, score) in enumerate(rankings[:20], 1):
            winner_class = "winner" if i == 1 else ""
            html += f'<tr class="{winner_class}"><td>{i}</td><td>{agent}</td><td>{score:.3f}</td><td>{data.get("elo_ratings", {}).get(agent, "N/A")}</td></tr>\n'

        html += """</table>
        </div>
    </div>
</body>
</html>"""
        return html

    def _generate_comparison_html(self, runs_data: List[Dict[str, Any]]) -> str:
        """Generate HTML for comparison report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Run Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2980b9; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Run Comparison</h1>
        </div>
        <table>
            <tr><th>Run</th><th>Best Fitness</th><th>Generations</th><th>Duration</th></tr>"""

        for run in runs_data:
            html += f'<tr><td>{run.get("name", run.get("id", "Unknown"))}</td><td>{run.get("best_fitness", 0):.3f}</td><td>{run.get("generations", 0)}</td><td>{run.get("duration", 0):.1f}s</td></tr>\n'

        html += """</table>
    </div>
</body>
</html>"""
        return html

    # =============================================================================
    # Markdown Generator
    # =============================================================================

    def _generate_markdown(self, data: Dict[str, Any], report_type: str) -> str:
        """Generate Markdown report."""
        md = f"# {report_type.title()} Report\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if report_type == ReportType.TRAINING:
            md += f"## Summary\n\n"
            md += f"- **Best Fitness**: {data.get('best_fitness', 0):.4f}\n"
            md += f"- **Mean Fitness**: {data.get('mean_fitness', 0):.4f}\n"
            md += f"- **Generations**: {data.get('generations', 0)}\n"
            md += f"- **Diversity**: {data.get('diversity', 0):.4f}\n"

        elif report_type == ReportType.EXPERIMENT:
            summary = data.get("summary", {})
            md += f"## Experiment: {data.get('name', 'Unknown')}\n\n"
            md += f"- **Total Runs**: {summary.get('total_runs', 0)}\n"
            md += f"- **Completed**: {summary.get('completed_runs', 0)}\n"
            md += f"- **Duration**: {summary.get('duration_hours', 0):.2f} hours\n"

        md += "\n---\n"
        return md
