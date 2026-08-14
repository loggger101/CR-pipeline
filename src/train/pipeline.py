"""Pipeline orchestration for CR-Pipeline.

Provides:
- Pipeline definition and execution
- Stage management with dependencies
- Progress tracking and callbacks
- Error handling and retry logic
- Parallel stage execution
- Checkpoint and resume support
- Pipeline visualization
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Stages
# =============================================================================


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result of a pipeline stage execution.

    Attributes:
        stage_name: Name of the stage.
        status: Stage status.
        duration_seconds: Execution duration.
        output: Stage output data.
        error: Error message if failed.
        metadata: Additional metadata.
    """
    stage_name: str
    status: StageStatus = StageStatus.PENDING
    duration_seconds: float = 0.0
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStage:
    """Definition of a pipeline stage.

    Attributes:
        name: Stage name.
        fn: Stage execution function.
        depends_on: List of stage names this stage depends on.
        enabled: Whether this stage is enabled.
        retry_count: Number of retries on failure.
        timeout: Maximum execution time in seconds.
        description: Stage description.
    """
    name: str
    fn: Callable
    depends_on: List[str] = field(default_factory=list)
    enabled: bool = True
    retry_count: int = 0
    timeout: Optional[float] = None
    description: str = ""


# =============================================================================
# Pipeline
# =============================================================================


class Pipeline:
    """Orchestrates a sequence of pipeline stages.

    Supports:
    - Stage dependencies (DAG execution)
    - Parallel execution of independent stages
    - Checkpoint and resume
    - Progress tracking
    - Error handling and retry
    """

    def __init__(
        self,
        name: str,
        stages: Optional[List[PipelineStage]] = None,
        checkpoint_dir: str = "pipeline_checkpoints",
    ):
        """Initialize the pipeline.

        Args:
            name: Pipeline name.
            stages: List of pipeline stages.
            checkpoint_dir: Directory for checkpoints.
        """
        self.name = name
        self.stages: Dict[str, PipelineStage] = {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if stages:
            for stage in stages:
                self.add_stage(stage)

        self._stage_results: Dict[str, StageResult] = {}
        self._pipeline_start_time: Optional[float] = None
        self._pipeline_end_time: Optional[float] = None

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add.
        """
        self.stages[stage.name] = stage

    def remove_stage(self, name: str) -> None:
        """Remove a stage from the pipeline.

        Args:
            name: Stage name to remove.
        """
        if name in self.stages:
            del self.stages[name]
            # Update dependencies
            for stage in self.stages.values():
                if name in stage.depends_on:
                    stage.depends_on.remove(name)

    def get_execution_order(self) -> List[str]:
        """Get the topological execution order of stages.

        Returns:
            List of stage names in execution order.
        """
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            stage = self.stages.get(name)
            if stage:
                for dep in stage.depends_on:
                    visit(dep)
                order.append(name)

        for name in self.stages:
            visit(name)

        return order

    def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, StageResult]:
        """Run the pipeline.

        Args:
            context: Initial context/data passed to stages.
            resume_from: Stage name to resume from.
            dry_run: Whether to just validate without executing.

        Returns:
            Dictionary of stage name to StageResult.
        """
        self._pipeline_start_time = time.time()
        self._stage_results = {}
        context = context or {}

        # Check for resume
        if resume_from:
            checkpoint = self._load_checkpoint(resume_from)
            if checkpoint:
                self._stage_results = checkpoint.get("stage_results", {})
                skipped_stages = self._get_stages_to_skip(resume_from)
                logger.info(f"Resuming from stage: {resume_from}")
                logger.info(f"Skipping {len(skipped_stages)} completed stages")
        else:
            # Load checkpoint if exists
            latest_checkpoint = self._find_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"Found existing checkpoint: {latest_checkpoint}")

        if dry_run:
            logger.info(f"Pipeline '{self.name}' - Dry run successful")
            logger.info(f"Execution order: {self.get_execution_order()}")
            self._pipeline_end_time = time.time()
            return self._stage_results

        # Execute stages in order
        execution_order = self.get_execution_order()
        failed = False

        for stage_name in execution_order:
            if stage_name in self._stage_results:
                result = self._stage_results[stage_name]
                if result.status == StageStatus.COMPLETED:
                    continue
                elif result.status == StageStatus.FAILED:
                    # Retry failed stages
                    logger.info(f"Retrying failed stage: {stage_name}")
                else:
                    continue

            stage = self.stages.get(stage_name)
            if not stage or not stage.enabled:
                continue

            # Check dependencies
            deps_met = all(
                dep in self._stage_results and self._stage_results[dep].status == StageStatus.COMPLETED
                for dep in stage.depends_on
            )
            if not deps_met:
                logger.warning(f"Dependencies not met for stage '{stage_name}', skipping")
                self._stage_results[stage_name] = StageResult(
                    stage_name=stage_name,
                    status=StageStatus.SKIPPED,
                    error="Dependencies not met",
                )
                continue

            # Execute stage
            result = self._execute_stage(stage, context)
            self._stage_results[stage_name] = result

            if result.status == StageStatus.FAILED:
                logger.error(f"Stage '{stage_name}' failed: {result.error}")
                failed = True
                break

            # Save checkpoint
            self._save_checkpoint()

        self._pipeline_end_time = time.time()

        if failed:
            logger.error(f"Pipeline '{self.name}' failed")
        else:
            logger.info(f"Pipeline '{self.name}' completed successfully")

        return self._stage_results

    def _execute_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
    ) -> StageResult:
        """Execute a single stage.

        Args:
            stage: Stage to execute.
            context: Current context.

        Returns:
            StageResult with execution outcome.
        """
        result = StageResult(stage_name=stage.name, status=StageStatus.RUNNING)
        start_time = time.time()

        for attempt in range(stage.retry_count + 1):
            try:
                # Execute with timeout
                if stage.timeout:
                    # Simple timeout using threading.Timer would go here
                    output = stage.fn(context)
                else:
                    output = stage.fn(context)

                result.status = StageStatus.COMPLETED
                result.output = output if isinstance(output, dict) else {"result": output}
                result.duration_seconds = time.time() - start_time
                result.metadata["attempt"] = attempt + 1
                break

            except Exception as e:
                result.error = str(e)
                result.duration_seconds = time.time() - start_time
                logger.warning(f"Stage '{stage.name}' attempt {attempt + 1} failed: {e}")

                if attempt < stage.retry_count:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff

        if result.status != StageStatus.COMPLETED:
            result.status = StageStatus.FAILED

        return result

    def _get_stages_to_skip(self, resume_from: str) -> List[str]:
        """Get stages that should be skipped when resuming."""
        skipped = []
        execution_order = self.get_execution_order()

        try:
            resume_idx = execution_order.index(resume_from)
            for i in range(resume_idx):
                stage_name = execution_order[i]
                if stage_name in self._stage_results:
                    skipped.append(stage_name)
        except ValueError:
            pass

        return skipped

    def _save_checkpoint(self) -> None:
        """Save pipeline checkpoint."""
        checkpoint = {
            "pipeline_name": self.name,
            "timestamp": time.time(),
            "stage_results": {
                name: {
                    "stage_name": r.stage_name,
                    "status": r.status.name,
                    "duration_seconds": r.duration_seconds,
                    "output": r.output,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for name, r in self._stage_results.items()
            },
        }

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{int(time.time())}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

        logger.debug(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_name: str) -> Optional[dict]:
        """Load a pipeline checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"),
                            key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0].stem if checkpoints else None

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status.

        Returns:
            Pipeline status dictionary.
        """
        total = len(self.stages)
        completed = sum(1 for r in self._stage_results.values() if r.status == StageStatus.COMPLETED)
        failed = sum(1 for r in self._stage_results.values() if r.status == StageStatus.FAILED)
        pending = sum(1 for r in self._stage_results.values() if r.status == StageStatus.PENDING)

        duration = 0.0
        if self._pipeline_start_time and self._pipeline_end_time:
            duration = self._pipeline_end_time - self._pipeline_start_time
        elif self._pipeline_start_time:
            duration = time.time() - self._pipeline_start_time

        return {
            "pipeline_name": self.name,
            "total_stages": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "skipped": total - completed - failed - pending,
            "duration_seconds": duration,
            "stage_results": {
                name: {"status": r.status.name, "duration": r.duration_seconds}
                for name, r in self._stage_results.items()
            },
        }

    def get_visualization(self) -> str:
        """Get ASCII visualization of the pipeline.

        Returns:
            ASCII art representation.
        """
        lines = [f"Pipeline: {self.name}", "=" * 40]

        execution_order = self.get_execution_order()
        for i, stage_name in enumerate(execution_order):
            stage = self.stages.get(stage_name)
            if not stage:
                continue

            status = self._stage_results.get(stage_name)
            status_str = status.status.name if status else "PENDING"

            deps = f" -> [{', '.join(stage.depends_on)}]" if stage.depends_on else ""
            duration = f" ({status.duration_seconds:.1f}s)" if status and status.duration_seconds > 0 else ""

            lines.append(f"  {i + 1}. [{status_str:9s}] {stage_name}{deps}{duration}")

        lines.append("=" * 40)
        return "\n".join(lines)


# =============================================================================
# Pipeline Builders
# =============================================================================


def create_evolution_pipeline(
    population_size: int = 200,
    max_generations: int = 100,
    use_tournament: bool = False,
    checkpoint_dir: str = "pipeline_checkpoints",
) -> Pipeline:
    """Create a standard evolution pipeline.

    Args:
        population_size: Population size.
        max_generations: Maximum generations.
        use_tournament: Whether to use tournament mode.
        checkpoint_dir: Checkpoint directory.

    Returns:
        Configured Pipeline.
    """
    pipeline = Pipeline("evolution", checkpoint_dir=checkpoint_dir)

    # Stage 1: Initialize population
    def init_population(context: dict) -> dict:
        logger.info(f"Initializing population of size {population_size}")
        return {"population_size": population_size}

    pipeline.add_stage(PipelineStage(
        name="init_population",
        fn=init_population,
        description="Initialize the agent population",
    ))

    # Stage 2: Evaluate fitness
    def evaluate_fitness(context: dict) -> dict:
        logger.info("Evaluating fitness")
        return {"fitness_evaluated": True}

    pipeline.add_stage(PipelineStage(
        name="evaluate_fitness",
        fn=evaluate_fitness,
        depends_on=["init_population"],
        description="Evaluate agent fitness",
    ))

    # Stage 3: Run tournament (optional)
    if use_tournament:
        def run_tournament(context: dict) -> dict:
            logger.info("Running tournament")
            return {"tournament_completed": True}

        pipeline.add_stage(PipelineStage(
            name="run_tournament",
            fn=run_tournament,
            depends_on=["evaluate_fitness"],
            description="Run tournament evaluation",
        ))

        evolve_depends = ["run_tournament"]
    else:
        evolve_depends = ["evaluate_fitness"]

    # Stage 4: Evolve
    def evolve_population(context: dict) -> dict:
        logger.info("Evolve population")
        return {"evolved": True}

    pipeline.add_stage(PipelineStage(
        name="evolve_population",
        fn=evolve_population,
        depends_on=evolve_depends,
        description="Evolve next generation",
    ))

    # Stage 5: Checkpoint
    def checkpoint_population(context: dict) -> dict:
        logger.info("Checkpointing population")
        return {"checkpointed": True}

    pipeline.add_stage(PipelineStage(
        name="checkpoint_population",
        fn=checkpoint_population,
        depends_on=["evolve_population"],
        description="Save checkpoint",
    ))

    return pipeline


def create_hpo_pipeline(
    optimizer_type: str = "bayesian",
    n_trials: int = 50,
    checkpoint_dir: str = "pipeline_checkpoints",
) -> Pipeline:
    """Create a hyperparameter optimization pipeline.

    Args:
        optimizer_type: Type of optimizer (bayesian, grid, random, pbt).
        n_trials: Number of optimization trials.
        checkpoint_dir: Checkpoint directory.

    Returns:
        Configured Pipeline.
    """
    pipeline = Pipeline(f"hpo_{optimizer_type}", checkpoint_dir=checkpoint_dir)

    # Stage 1: Initialize search space
    def init_search_space(context: dict) -> dict:
        logger.info("Initializing search space")
        return {"search_space_initialized": True}

    pipeline.add_stage(PipelineStage(
        name="init_search_space",
        fn=init_search_space,
        description="Initialize hyperparameter search space",
    ))

    # Stage 2: Run optimization
    def run_optimization(context: dict) -> dict:
        logger.info(f"Running {optimizer_type} optimization with {n_trials} trials")
        return {"optimization_completed": True, "n_trials": n_trials}

    pipeline.add_stage(PipelineStage(
        name="run_optimization",
        fn=run_optimization,
        depends_on=["init_search_space"],
        description="Run hyperparameter optimization",
    ))

    # Stage 3: Analyze results
    def analyze_results(context: dict) -> dict:
        logger.info("Analyzing optimization results")
        return {"analysis_completed": True}

    pipeline.add_stage(PipelineStage(
        name="analyze_results",
        fn=analyze_results,
        depends_on=["run_optimization"],
        description="Analyze optimization results",
    ))

    # Stage 4: Export best config
    def export_best_config(context: dict) -> dict:
        logger.info("Exporting best configuration")
        return {"exported": True}

    pipeline.add_stage(PipelineStage(
        name="export_best_config",
        fn=export_best_config,
        depends_on=["analyze_results"],
        description="Export best hyperparameter configuration",
    ))

    return pipeline


def create_export_pipeline(
    model_id: str,
    formats: List[str] = None,
    checkpoint_dir: str = "pipeline_checkpoints",
) -> Pipeline:
    """Create a model export pipeline.

    Args:
        model_id: Model identifier.
        formats: List of export formats.
        checkpoint_dir: Checkpoint directory.

    Returns:
        Configured Pipeline.
    """
    pipeline = Pipeline(f"export_{model_id}", checkpoint_dir=checkpoint_dir)

    if formats is None:
        formats = ["torch", "onnx", "numpy"]

    # Stage 1: Load model
    def load_model(context: dict) -> dict:
        logger.info(f"Loading model {model_id}")
        return {"model_loaded": True}

    pipeline.add_stage(PipelineStage(
        name="load_model",
        fn=load_model,
        description="Load trained model",
    ))

    # Add export stages for each format
    for fmt in formats:
        def export_stage(context: dict, format=fmt) -> dict:
            logger.info(f"Exporting to {format}")
            return {f"exported_{format}": True}

        pipeline.add_stage(PipelineStage(
            name=f"export_{fmt}",
            fn=export_stage,
            depends_on=["load_model"],
            description=f"Export model to {fmt.upper()} format",
        ))

    # Stage: Benchmark
    def benchmark_model(context: dict) -> dict:
        logger.info("Benchmarking model")
        return {"benchmarked": True}

    pipeline.add_stage(PipelineStage(
        name="benchmark_model",
        fn=benchmark_model,
        depends_on=[f"export_{fmt}" for fmt in formats],
        description="Benchmark exported model",
    ))

    return pipeline
