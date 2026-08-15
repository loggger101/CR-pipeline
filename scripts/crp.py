#!/usr/bin/env python3
"""CR-Pipeline CLI - Command line interface for the framework.

Usage:
    crp <command> [options]

Commands:
    train         Start training
    tournament    Run tournament evaluation
    hpo           Run hyperparameter optimization
    export        Export trained models
    report        Generate reports
    dashboard     Launch visualization dashboard
    compare       Compare runs
    experiments   List/manage experiments
    pipelines     List/run pipelines
    search        Run architecture search
    benchmark     Benchmark models

Examples:
    crp train --max-gens 100 --population-size 200
    crp tournament --format round_robin --matches 4
    crp hpo --optimizer bayesian --trials 50
    crp export --model best --formats torch,onnx,numpy
    crp report --experiment exp_123 --type training
    crp compare --runs run_1 run_2 run_3
    crp dashboard --runs-dir runs
    crp pipelines --list
    crp pipelines --run evolution
    crp search --generations 50 --population 20
    crp benchmark --model my_model.pt --input-shape 1,8,6,16
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serialization import load_checkpoint  # noqa: E402

logger = logging.getLogger("crp")


def setup_logging(verbose: bool = False) -> None:
    """Set up logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Handle train command."""
    from src.train import TrainingConfig, EvolutionTrainer

    config = TrainingConfig(
        population_size=args.population_size,
        elite_count=args.elite_count,
        max_generations=args.max_gens,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_std=args.mutation_std,
        tournament_mode=args.tournament,
        tournament_format=args.tournament_format,
        tournament_matches=args.tournament_matches,
        num_workers=args.workers,
        runs_dir=args.runs_dir,
        resume_from=args.resume,
        seed=args.seed,
        monitor_resources=args.monitor,
        enable_alerts=args.alerts,
        enable_registry=args.registry,
        collect_matches=args.collect_matches,
    )

    trainer = EvolutionTrainer(config)
    logger.info(f"Starting training: {config.max_generations} generations")
    logger.info(f"Population: {config.population_size}, Elite: {config.elite_count}")

    try:
        trainer.train()
        logger.info("Training completed successfully")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


def cmd_tournament(args: argparse.Namespace) -> int:
    """Handle tournament command."""
    from src.train import FitnessEvaluator, TournamentFormat

    format_map = {
        "round_robin": TournamentFormat.ROUND_ROBIN,
        "single_elim": TournamentFormat.SINGLE_ELIMINATION,
        "double_elim": TournamentFormat.DOUBLE_ELIMINATION,
        "league": TournamentFormat.LEAGUE,
    }

    tournament_format = format_map.get(args.format, TournamentFormat.ROUND_ROBIN)

    evaluator = FitnessEvaluator(num_workers=args.workers, matches_per_agent=args.matches)

    # Load population
    population_path = Path(args.population)
    import torch
    checkpoint = load_checkpoint(str(population_path))

    if isinstance(checkpoint, dict) and "agents" in checkpoint:
        weights_list = [np.array(a["weights"]) for a in checkpoint["agents"]]
    elif isinstance(checkpoint, dict) and "weights" in checkpoint:
        weights_list = [np.array(checkpoint["weights"])]
    else:
        weights_list = [np.array(checkpoint)]

    agent_ids = [f"agent_{i}" for i in range(len(weights_list))]

    logger.info(f"Running {tournament_format.name} tournament with {len(agent_ids)} agents")

    result = evaluator.run_tournament(
        agent_ids=agent_ids,
        weights_list=weights_list,
        format=tournament_format,
        matches_per_pair=args.matches,
        seed=args.seed,
        generation=args.generation,
    )

    print("\n" + result.summary())

    evaluator.shutdown()
    return 0


def cmd_hpo(args: argparse.Namespace) -> int:
    """Handle HPO command."""
    from src.train import (
        BayesianOptimizer,
        GridSearchOptimizer,
        RandomSearchOptimizer,
        ParamSpace,
        ParamType,
        get_default_evolution_search_space,
    )

    search_space = get_default_evolution_search_space()

    # Define objective function
    def objective(params):
        # Simulated objective
        pop_size = params.get("population_size", 200)
        mutation_rate = params.get("mutation_rate", 0.05)
        crossover_rate = params.get("crossover_rate", 0.7)

        # Simulated fitness (higher is better)
        fitness = (
            10.0
            - abs(pop_size - 200) * 0.01
            - abs(mutation_rate - 0.05) * 10
            - abs(crossover_rate - 0.7) * 5
            + np.random.randn() * 0.1
        )
        return fitness

    if args.optimizer == "bayesian":
        optimizer = BayesianOptimizer(
            param_spaces=search_space,
            n_initial=args.initial,
            n_iterations=args.trials,
            seed=args.seed,
        )
    elif args.optimizer == "grid":
        optimizer = GridSearchOptimizer(
            param_spaces=search_space,
            grid_points=args.grid_points,
        )
    elif args.optimizer == "random":
        optimizer = RandomSearchOptimizer(
            param_spaces=search_space,
            n_trials=args.trials,
            patience=args.patience,
            seed=args.seed,
        )
    else:
        logger.error(f"Unknown optimizer: {args.optimizer}")
        return 1

    logger.info(f"Running {args.optimizer} optimization with {args.trials} trials")

    result = optimizer.optimize(objective, verbose=True)

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Best Parameters: {json.dumps(result.best_params, indent=2)}")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Evaluations: {result.n_evaluations}")
    print(f"Time: {result.optimization_time:.1f}s")
    print("=" * 60)

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Handle export command."""
    from src.deploy import ModelExporter, ModelMetadata, ModelFormat

    # Load model
    model_path = Path(args.model)
    import torch
    checkpoint = load_checkpoint(str(model_path))

    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        weights = np.array(checkpoint["weights"])
    else:
        weights = np.array(checkpoint)

    # Create exporter
    exporter = ModelExporter(output_dir=args.output_dir)

    formats = {
        "torch": ModelFormat.TORCH,
        "torchscript": ModelFormat.TORCHSCRIPT,
        "onnx": ModelFormat.ONNX,
        "numpy": ModelFormat.NUMPY,
        "json": ModelFormat.JSON,
        "pickle": ModelFormat.PICKLE,
    }

    metadata = ModelMetadata(
        model_id=args.model_id or Path(args.model).stem,
        version=args.version,
        format=ModelFormat.TORCH,  # Default
        architecture=args.architecture,
        input_shape=args.input_shape,
        output_shape=args.output_shape,
    )

    exported = []
    for fmt in args.formats.split(","):
        fmt = fmt.strip()
        if fmt in formats:
            path = exporter.export_model(weights, metadata, formats[fmt])
            exported.append(path)
            logger.info(f"Exported to {path}")

    print(f"\nExported {len(exported)} models:")
    for path in exported:
        print(f"  - {path}")

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Handle report command."""
    from src.train import ExperimentTracker
    from src.viz import ReportGenerator

    tracker = ExperimentTracker(tracking_dir=args.experiments_dir)

    if args.experiment:
        exp = tracker.get_experiment(args.experiment)
        if not exp:
            logger.error(f"Experiment {args.experiment} not found")
            return 1

        reporter = ReportGenerator(output_dir=args.output_dir)

        if args.type == "training":
            report = tracker.generate_report(args.experiment, args.metric, args.output)
        elif args.type == "experiment":
            report = reporter.generate_experiment_report(
                exp.get_summary(),
                args.output or "experiment_report.html",
            )
        else:
            report = tracker.generate_report(args.experiment, args.metric, args.output)

        print(f"Report generated: {report}")
    else:
        # List experiments
        experiments = tracker.list_experiments()
        print(f"\nExperiments ({len(experiments)}):")
        for exp in experiments:
            print(f"  {exp.experiment_id}: {exp.name} ({len(exp.runs)} runs)")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Handle compare command."""
    from src.train import ExperimentTracker

    tracker = ExperimentTracker(tracking_dir=args.experiments_dir)

    # Load runs
    runs = []
    for run_id in args.runs:
        run = tracker.get_run(run_id)
        if run:
            runs.append(run)

    if not runs:
        logger.error("No runs found to compare")
        return 1

    # Compare
    run_ids = [r.run_id for r in runs]
    comparison = tracker.compare_runs(run_ids, args.metric)

    print("\n" + "=" * 60)
    print("RUN COMPARISON")
    print("=" * 60)
    print(f"Metric: {args.metric}")
    print(f"Best Run: {comparison.get('best_run', 'N/A')}")
    print(f"Best Value: {comparison.get('best_value', 'N/A')}")
    print(f"Avg Best: {comparison.get('avg_best', 'N/A')}")
    print(f"Spread: {comparison.get('spread', 'N/A')}")
    print("-" * 60)

    for run_id, data in comparison.get("runs", {}).items():
        print(f"  {run_id}: best={data.get('best', 'N/A')}, steps={data.get('steps', 0)}")

    print("=" * 60)

    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Handle dashboard command."""
    from src.viz import run_advanced_dashboard

    print("Launching advanced dashboard...")
    run_advanced_dashboard(runs_dir=args.runs_dir, refresh_interval=args.refresh)
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    """Handle experiments command."""
    from src.train import ExperimentTracker

    tracker = ExperimentTracker(tracking_dir=args.experiments_dir)

    if args.action == "list":
        experiments = tracker.list_experiments(tag=args.tag)
        print(f"\nExperiments ({len(experiments)}):")
        for exp in experiments:
            summary = exp.get_summary()
            print(f"  {exp.experiment_id}: {exp.name}")
            print(f"    Runs: {summary['total_runs']} (completed: {summary['completed_runs']})")
            print(f"    Duration: {summary['duration_hours']:.2f}h")
            print(f"    Tags: {', '.join(exp.tags)}")

    elif args.action == "create":
        exp = tracker.create_experiment(args.name, args.description, args.tags)
        print(f"Created experiment: {exp.experiment_id}")

    elif args.action == "summary":
        if args.experiment:
            exp = tracker.get_experiment(args.experiment)
            if exp:
                summary = exp.get_summary()
                print(json.dumps(summary, indent=2, default=str))
        else:
            summaries = tracker.get_experiment_summaries()
            print(json.dumps(summaries, indent=2, default=str))

    return 0


def cmd_pipelines(args: argparse.Namespace) -> int:
    """Handle pipelines command."""
    from src.train import create_evolution_pipeline, create_hpo_pipeline

    if args.action == "list":
        print("\nAvailable pipelines:")
        print("  evolution   - Standard evolution training")
        print("  hpo         - Hyperparameter optimization")
        print("  export      - Model export and benchmarking")

    elif args.action == "run":
        if args.name == "evolution":
            pipeline = create_evolution_pipeline(
                population_size=args.pop_size,
                max_generations=args.max_gens,
                use_tournament=args.tournament,
            )
            print(f"Running evolution pipeline...")
            result = pipeline.run()
            status = pipeline.get_status()
            print(f"Completed: {status['completed']}/{status['total_stages']} stages")
            print(f"Duration: {status['duration_seconds']:.1f}s")

        elif args.name == "hpo":
            pipeline = create_hpo_pipeline(
                optimizer_type=args.optimizer_type,
                n_trials=args.trials,
            )
            print(f"Running HPO pipeline...")
            result = pipeline.run()
            status = pipeline.get_status()
            print(f"Completed: {status['completed']}/{status['total_stages']} stages")

        else:
            print(f"Unknown pipeline: {args.name}")
            return 1

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Handle architecture search command."""
    from src.models import ArchitectureEvolver, ArchitectureRegistry

    evolver = ArchitectureEvolver(
        population_size=args.population,
        elite_fraction=args.elite_fraction,
        seed=args.seed,
    )
    evolver.initialize_population()

    # Simple fitness function
    def fitness_fn(arch):
        return len(arch.layers) * 0.1 + np.random.randn() * 0.01

    print(f"Running architecture search for {args.generations} generations...")
    best_arch, info = evolver.evolve_generation(
        fitness_fn=fitness_fn,
        n_generations=args.generations,
        verbose=True,
    )

    if best_arch:
        print(f"\nBest Architecture:")
        print(f"  Name: {best_arch.name}")
        print(f"  Layers: {len(best_arch.layers)}")
        print(f"  Parameters: {best_arch.num_parameters}")
        print(f"  Final Fitness: {info['final_fitness']:.4f}")

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Handle benchmark command."""
    from src.deploy import InferenceBenchmarker

    # Load model
    model_path = Path(args.model)
    import torch
    checkpoint = load_checkpoint(str(model_path))

    if isinstance(checkpoint, dict) and "weights" in checkpoint:
        weights = np.array(checkpoint["weights"])
    else:
        weights = np.array(checkpoint)

    # Create simple model function
    def model_fn(x):
        return x @ weights.T[:x.shape[1]] if x.shape[1] <= weights.shape[0] else x @ weights[:x.shape[1]].T

    # Parse input shape
    input_shape = tuple(int(x) for x in args.input_shape.split(","))

    benchmarker = InferenceBenchmarker()
    results = benchmarker.benchmark(
        model_fn=model_fn,
        input_shape=input_shape,
        n_runs=args.n_runs,
        warmup_runs=args.warmup,
    )

    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Mean Latency: {results['mean_latency_ms']:.2f} ms")
    print(f"Median Latency: {results['median_latency_ms']:.2f} ms")
    print(f"P95 Latency: {results['p95_latency_ms']:.2f} ms")
    print(f"P99 Latency: {results['p99_latency_ms']:.2f} ms")
    print(f"Min Latency: {results['min_latency_ms']:.2f} ms")
    print(f"Max Latency: {results['max_latency_ms']:.2f} ms")
    print(f"Mean Throughput: {results['mean_throughput']:.1f} samples/sec")
    print(f"Total Time: {results['total_time_s']:.2f}s")
    print("=" * 60)

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="crp",
        description="CR-Pipeline CLI - Evolutionary Neural Network Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  crp train --max-gens 100 --population-size 200
  crp tournament --population runs/best/checkpoint.pt
  crp hpo --optimizer bayesian --trials 50
  crp export --model best.pt --formats torch,onnx,numpy
  crp report --experiment exp_123
  crp compare --runs run_1 run_2 run_3
  crp dashboard --runs-dir runs
  crp pipelines --list
  crp search --generations 50
  crp benchmark --model model.pt --input-shape 1,8,6,16
        """,
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--version", "-V", action="version", version="CR-Pipeline 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--max-gens", "-g", type=int, default=100)
    train_parser.add_argument("--population-size", "-p", type=int, default=200)
    train_parser.add_argument("--elite-count", "-e", type=int, default=10)
    train_parser.add_argument("--crossover-rate", type=float, default=0.7)
    train_parser.add_argument("--mutation-rate", type=float, default=0.05)
    train_parser.add_argument("--mutation-std", type=float, default=0.1)
    train_parser.add_argument("--tournament", action="store_true", help="Use tournament mode")
    train_parser.add_argument("--tournament-format", default="round_robin")
    train_parser.add_argument("--tournament-matches", type=int, default=4)
    train_parser.add_argument("--workers", "-w", type=int, default=4)
    train_parser.add_argument("--runs-dir", "-r", default="runs")
    train_parser.add_argument("--resume", type=str, default=None)
    train_parser.add_argument("--seed", "-s", type=int, default=42)
    # Monitoring & registry flags
    train_parser.add_argument("--monitor", action="store_true", help="Enable resource monitoring")
    train_parser.add_argument("--alerts", action="store_true", help="Enable alerting")
    train_parser.add_argument("--registry", action="store_true", help="Enable model registry")
    train_parser.add_argument("--collect-matches", action="store_true", help="Collect match data")

    # Tournament command
    tour_parser = subparsers.add_parser("tournament", help="Run tournament evaluation")
    tour_parser.add_argument("--population", "-p", required=True, help="Population checkpoint path")
    tour_parser.add_argument("--format", "-f", default="round_robin",
                             choices=["round_robin", "single_elim", "double_elim", "league"])
    tour_parser.add_argument("--matches", "-m", type=int, default=4)
    tour_parser.add_argument("--workers", "-w", type=int, default=4)
    tour_parser.add_argument("--seed", "-s", type=int, default=42)
    tour_parser.add_argument("--generation", "-g", type=int, default=0)

    # HPO command
    hpo_parser = subparsers.add_parser("hpo", help="Run hyperparameter optimization")
    hpo_parser.add_argument("--optimizer", "-o", default="bayesian",
                            choices=["bayesian", "grid", "random"])
    hpo_parser.add_argument("--trials", "-t", type=int, default=50)
    hpo_parser.add_argument("--initial", type=int, default=10)
    hpo_parser.add_argument("--grid-points", type=int, default=5)
    hpo_parser.add_argument("--patience", type=int, default=5)
    hpo_parser.add_argument("--seed", "-s", type=int, default=42)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained models")
    export_parser.add_argument("--model", "-m", required=True, help="Model checkpoint path")
    export_parser.add_argument("--formats", "-f", default="torch,numpy,json",
                               help="Export formats (comma-separated)")
    export_parser.add_argument("--model-id", type=str, default=None)
    export_parser.add_argument("--version", "-V", default="1.0")
    export_parser.add_argument("--architecture", default="custom")
    export_parser.add_argument("--input-shape", type=str, default="1,8,6,16")
    export_parser.add_argument("--output-shape", type=str, default="1,100")
    export_parser.add_argument("--output-dir", "-o", default="exports")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--experiment", "-e", type=str, default=None)
    report_parser.add_argument("--type", "-t", default="training",
                               choices=["training", "experiment", "tournament"])
    report_parser.add_argument("--metric", default="best_fitness")
    report_parser.add_argument("--output", "-o", type=str, default=None)
    report_parser.add_argument("--experiments-dir", default="experiment_tracking")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare runs")
    compare_parser.add_argument("runs", nargs="+", help="Run IDs to compare")
    compare_parser.add_argument("--metric", "-m", default="best_fitness")
    compare_parser.add_argument("--experiments-dir", default="experiment_tracking")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch visualization dashboard")
    dash_parser.add_argument("--runs-dir", "-r", default="runs")
    dash_parser.add_argument("--refresh", type=int, default=10)

    # Experiments command
    exp_parser = subparsers.add_parser("experiments", help="Manage experiments")
    exp_parser.add_argument("action", choices=["list", "create", "summary"])
    exp_parser.add_argument("--name", "-n", type=str, default=None)
    exp_parser.add_argument("--description", "-d", type=str, default="")
    exp_parser.add_argument("--tags", "-t", nargs="*", default=[])
    exp_parser.add_argument("--experiment", "-e", type=str, default=None)
    exp_parser.add_argument("--tag", type=str, default=None)
    exp_parser.add_argument("--experiments-dir", default="experiment_tracking")

    # Pipelines command
    pipe_parser = subparsers.add_parser("pipelines", help="Manage pipelines")
    pipe_parser.add_argument("action", choices=["list", "run"])
    pipe_parser.add_argument("--name", "-n", type=str, default=None)
    pipe_parser.add_argument("--pop-size", type=int, default=200)
    pipe_parser.add_argument("--max-gens", type=int, default=100)
    pipe_parser.add_argument("--tournament", action="store_true")
    pipe_parser.add_argument("--optimizer-type", default="bayesian")
    pipe_parser.add_argument("--trials", type=int, default=50)

    # Search command
    search_parser = subparsers.add_parser("search", help="Run architecture search")
    search_parser.add_argument("--generations", "-g", type=int, default=50)
    search_parser.add_argument("--population", "-p", type=int, default=20)
    search_parser.add_argument("--elite-fraction", type=float, default=0.2)
    search_parser.add_argument("--seed", "-s", type=int, default=42)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark models")
    bench_parser.add_argument("--model", "-m", required=True)
    bench_parser.add_argument("--input-shape", type=str, default="1,8,6,16")
    bench_parser.add_argument("--n-runs", type=int, default=100)
    bench_parser.add_argument("--warmup", type=int, default=10)

    # Model command
    model_parser = subparsers.add_parser("models", help="Manage model registry")
    model_parser.add_argument("action", choices=["list", "promote", "archive", "compare", "stats"])
    model_parser.add_argument("--model-id", "-m", type=str, default=None)
    model_parser.add_argument("--stage", "-s", type=str, default=None, choices=["checkpoint", "candidate", "production", "archived"])
    model_parser.add_argument("--models", nargs="*", default=[])
    model_parser.add_argument("--registry-dir", default="runs/model_registry")
    model_parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    setup_logging(args.verbose)

    # Config generator command
    config_parser = subparsers.add_parser("config", help="Generate configurations")
    config_parser.add_argument("type", choices=["evolution", "simulation", "tournament"], help="Config type")
    config_parser.add_argument("--preset", "-p", default="standard", help="Preset name")
    config_parser.add_argument("--name", "-n", type=str, default=None, help="Output filename")
    config_parser.add_argument("--override", "-o", nargs="*", default=[], help="Key=value overrides")
    config_parser.add_argument("--output-dir", default="configs/generated")

    # Dispatch to command handler
    commands = {
        "train": cmd_train,
        "tournament": cmd_tournament,
        "hpo": cmd_hpo,
        "export": cmd_export,
        "report": cmd_report,
        "compare": cmd_compare,
        "dashboard": cmd_dashboard,
        "experiments": cmd_experiments,
        "pipelines": cmd_pipelines,
        "search": cmd_search,
        "benchmark": cmd_benchmark,
        "models": cmd_models,
        "config": cmd_config,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


def cmd_models(args: argparse.Namespace) -> int:
    """Handle model registry commands."""
    from src.registry import ModelRegistry, ModelStage

    registry = ModelRegistry(args.registry_dir)

    if args.action == "list":
        models = registry.get_best_models(limit=args.limit)
        if not models:
            logger.info("No models registered.")
            return 0
        logger.info(f"{'ID':<20} {'Version':<10} {'Fitness':>10} {'Stage':<12} {'Architecture':<15}")
        logger.info("-" * 70)
        for m in models:
            logger.info(f"{m.model_id:<20} {m.version:<10} {m.fitness:>10.3f} {m.stage.name:<12} {m.architecture:<15}")
        return 0

    elif args.action == "promote":
        if not args.model_id:
            logger.error("--model-id required for promote")
            return 1
        stage_map = {
            "checkpoint": ModelStage.CHECKPOINT,
            "candidate": ModelStage.CANDIDATE,
            "production": ModelStage.PRODUCTION,
            "archived": ModelStage.ARCHIVED,
        }
        stage = stage_map.get(args.stage, ModelStage.PRODUCTION)
        model = registry.promote_model(args.model_id, stage)
        logger.info(f"Promoted {model.model_id} to {stage.name}")
        return 0

    elif args.action == "archive":
        if not args.model_id:
            logger.error("--model-id required for archive")
            return 1
        model = registry.promote_model(args.model_id, ModelStage.ARCHIVED)
        logger.info(f"Archived {model.model_id}")
        return 0

    elif args.action == "compare":
        if not args.models or len(args.models) < 2:
            logger.error("--models requires at least 2 model IDs")
            return 1
        comparison = registry.compare_models(args.models)
        for model_id, data in comparison.items():
            logger.info(f"{model_id}: fitness={data['fitness']:.3f}, arch={data['architecture']}, stage={data['stage']}")
        return 0

    elif args.action == "stats":
        models = list(registry.models.values())
        if not models:
            logger.info("No models registered.")
            return 0
        fitnesses = [m.fitness for m in models]
        logger.info(f"Total models: {len(models)}")
        logger.info(f"Best fitness: {max(fitnesses):.3f}")
        logger.info(f"Mean fitness: {sum(fitnesses)/len(fitnesses):.3f}")
        logger.info(f"Production models: {len(registry.get_production_models())}")
        return 0

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Handle config generation command."""
    from src.config.generator import ConfigGenerator, ConfigPreset

    type_map = {
        "evolution": {
            "fast": ConfigPreset.EVOLUTION_FAST,
            "standard": ConfigPreset.EVOLUTION_STANDARD,
            "competitive": ConfigPreset.EVOLUTION_COMPETITIVE,
        },
        "simulation": {
            "quick": ConfigPreset.SIMULATION_QUICK,
            "standard": ConfigPreset.SIMULATION_STANDARD,
            "detailed": ConfigPreset.SIMULATION_DETAILED,
        },
        "tournament": {
            "round_robin": ConfigPreset.TOURNAMENT_ROUND_ROBIN,
            "elimination": ConfigPreset.TOURNAMENT_ELIMINATION,
            "league": ConfigPreset.TOURNAMENT_LEAGUE,
        },
    }

    preset_map = type_map.get(args.type, {})
    preset = preset_map.get(args.preset)
    if not preset:
        logger.error(f"Unknown preset '{args.preset}' for type '{args.type}'")
        logger.info(f"Available presets: {', '.join(preset_map.keys())}")
        return 1

    generator = ConfigGenerator(output_dir=args.output_dir)
    overrides = {}
    for ov in args.override:
        if '=' in ov:
            key, value = ov.split('=', 1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            overrides[key] = value

    config = generator.generate(preset, args.name, overrides if overrides else None)
    logger.info(f"Configuration generated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
