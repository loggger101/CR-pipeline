"""CR-Pipeline: Neural Network Models & Evolution.

Provides:
- Network architectures (CNN+LSTM, CNN+MLP)
- Evolutionary agent wrapper with action selection
- Population management with fitness tracking
- Genetic algorithm operators (selection, crossover, mutation)
"""

from .architecture import (
    create_cnn_lstm_agent,
    create_cnn_mlp_agent,
    AgentArchitecture,
    CNNLSTMAgent,
    CNNMLPAgent,
)
from .agent import EvolutionaryAgent, AgentConfig
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
    BlendCrossover,
    SinglePointCrossover,
    UniformCrossover,
    GaussianMutation,
    UniformMutation,
    AdaptiveMutation,
)

__all__ = [
    # Architecture
    "create_cnn_lstm_agent",
    "create_cnn_mlp_agent",
    "AgentArchitecture",
    "CNNLSTMAgent",
    "CNNMLPAgent",
    # Agent
    "EvolutionaryAgent",
    "AgentConfig",
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
    "BlendCrossover",
    "SinglePointCrossover",
    "UniformCrossover",
    "GaussianMutation",
    "UniformMutation",
    "AdaptiveMutation",
]
