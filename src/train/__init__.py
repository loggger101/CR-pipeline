"""CR-Pipeline: Training Pipeline.

Provides:
- EvolutionTrainer: Main training loop orchestrator
- FitnessEvaluator: Agent evaluation against opponents
- CheckpointManager: Save/load training state
- HyperparamsLoader: Configuration management with phase presets
- Hyperparameter optimization (Bayesian, Grid, Random, PBT)
- Experiment tracking and reporting
- Pipeline orchestration
"""

from .trainer import EvolutionTrainer, TrainingConfig
from .evaluator import FitnessEvaluator, TournamentResult, TournamentFormat, TournamentRunner, AgentTournamentStats, HeadToHeadRecord, TournamentBracket
from .checkpoint import CheckpointManager
from .hyperparams import (
    HyperparamsLoader,
    DEFAULT_EVOLUTION_CONFIG,
    DEFAULT_SIM_CONFIG,
    DEFAULT_LIVE_CONFIG,
    PHASE_CONFIGS,
)
from .hpo import (
    ParamSpace,
    ParamType,
    OptimizationResult,
    BayesianOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    PBTOptimizer,
    SensitivityAnalyzer,
    get_default_evolution_search_space,
    get_default_tournament_search_space,
)
from .experiment_tracking import (
    ExperimentTracker,
    Experiment,
    ExperimentRun,
    MetricPoint,
)
from .pipeline import (
    Pipeline,
    PipelineStage,
    StageStatus,
    StageResult,
    create_evolution_pipeline,
    create_hpo_pipeline,
    create_export_pipeline,
)

__all__ = [
    "EvolutionTrainer",
    "TrainingConfig",
    "FitnessEvaluator",
    "TournamentResult",
    "TournamentFormat",
    "TournamentRunner",
    "AgentTournamentStats",
    "HeadToHeadRecord",
    "TournamentBracket",
    "CheckpointManager",
    "HyperparamsLoader",
    "DEFAULT_EVOLUTION_CONFIG",
    "DEFAULT_SIM_CONFIG",
    "DEFAULT_LIVE_CONFIG",
    "PHASE_CONFIGS",
    # HPO
    "ParamSpace",
    "ParamType",
    "OptimizationResult",
    "BayesianOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "PBTOptimizer",
    "SensitivityAnalyzer",
    "get_default_evolution_search_space",
    "get_default_tournament_search_space",
    # Experiment Tracking
    "ExperimentTracker",
    "Experiment",
    "ExperimentRun",
    "MetricPoint",
    # Pipeline
    "Pipeline",
    "PipelineStage",
    "StageStatus",
    "StageResult",
    "create_evolution_pipeline",
    "create_hpo_pipeline",
    "create_export_pipeline",
]
