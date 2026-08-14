"""CR-Pipeline: Neural Network Models & Evolution.

Provides:
- Network architectures (CNN+LSTM, CNN+MLP, CNN+ResNet, CNN+Transformer)
- Evolutionary agent wrapper with action selection
- Population management with fitness tracking
- Genetic algorithm operators (selection, crossover, mutation)
- Diversity tracking and speciation

Enhanced with:
- Multiple exploration strategies (epsilon-greedy, Boltzmann, entropy)
- Diversity preservation and speciation
- Novelty search support
- Adaptive mutation rates
- Tournament elite selection
- Arithmetic crossover
"""

from .architecture import (
    create_cnn_lstm_agent,
    create_cnn_mlp_agent,
    create_cnn_resnet_agent,
    create_cnn_transformer_agent,
    create_cnn_cnn_mlp_agent,
    create_cnn_gru_agent,
    create_cnn_lstm_attention_agent,
    create_cnn_resnet_lstm_agent,
    create_cnn_transformer_lstm_agent,
    AgentArchitecture,
    CNNLSTMAgent,
    CNNMLPAgent,
    CNNResNetAgent,
    CNNTransformerAgent,
    CNNCNNMLPAgent,
    CNNGRUAgent,
    CNNLSTMAttentionAgent,
    CNNResNetLSTMAgent,
    CNNTransformerLSTMAgent,
)
from .agent import EvolutionaryAgent, AgentConfig, ExplorationStrategy
from .population import Population, AgentRecord
from .evolution import (
    EvolutionStrategy,
    EvolutionConfig,
    SelectionStrategy,
    CrossoverStrategy,
    MutationStrategy,
    TournamentSelection,
    RankSelection,
    RouletteSelection,
    TournamentEliteSelection,
    TournamentRankSelection,
    BlendCrossover,
    SinglePointCrossover,
    UniformCrossover,
    ArithmeticCrossover,
    GaussianMutation,
    UniformMutation,
    AdaptiveMutation,
    DiversityTracker,
    TournamentEvolutionStrategy,
)
from .architecture_search import (
    ArchitectureSearchSpace,
    ArchitectureEvolver,
    ArchitectureRegistry,
    ArchitectureConfig,
    LayerType,
)
from .ensemble import (
    EnsembleBuilder,
    EnsembleMethod,
    WeightAveragingEnsemble,
    GeometricMeanEnsemble,
    StackingEnsemble,
    DiversityMetric,
    EnsembleOptimizer,
    EnsembleResult,
)

__all__ = [
    # Architecture
    "create_cnn_lstm_agent",
    "create_cnn_mlp_agent",
    "create_cnn_resnet_agent",
    "create_cnn_transformer_agent",
    "create_cnn_cnn_mlp_agent",
    "create_cnn_gru_agent",
    "create_cnn_lstm_attention_agent",
    "create_cnn_resnet_lstm_agent",
    "create_cnn_transformer_lstm_agent",
    "AgentArchitecture",
    "CNNLSTMAgent",
    "CNNMLPAgent",
    "CNNResNetAgent",
    "CNNTransformerAgent",
    "CNNCNNMLPAgent",
    "CNNGRUAgent",
    "CNNLSTMAttentionAgent",
    "CNNResNetLSTMAgent",
    "CNNTransformerLSTMAgent",
    # Agent
    "EvolutionaryAgent",
    "AgentConfig",
    "ExplorationStrategy",
    # Population
    "Population",
    "AgentRecord",
    # Evolution
    "EvolutionStrategy",
    "EvolutionConfig",
    "SelectionStrategy",
    "CrossoverStrategy",
    "MutationStrategy",
    "TournamentSelection",
    "RankSelection",
    "RouletteSelection",
    "TournamentEliteSelection",
    "TournamentRankSelection",
    "TournamentEvolutionStrategy",
    "BlendCrossover",
    "SinglePointCrossover",
    "UniformCrossover",
    "ArithmeticCrossover",
    "GaussianMutation",
    "UniformMutation",
    "AdaptiveMutation",
    "DiversityTracker",
    # Architecture Search
    "ArchitectureSearchSpace",
    "ArchitectureEvolver",
    "ArchitectureRegistry",
    "ArchitectureConfig",
    "LayerType",
    # Ensemble
    "EnsembleBuilder",
    "EnsembleMethod",
    "WeightAveragingEnsemble",
    "GeometricMeanEnsemble",
    "StackingEnsemble",
    "DiversityMetric",
    "EnsembleOptimizer",
    "EnsembleResult",
]