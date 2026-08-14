"""Data augmentation for CR-Pipeline simulation.

Provides:
- Card deck augmentation strategies
- Arena state augmentation
- Opponent strategy variation
- Game condition augmentation
- Replay augmentation
- Training data generation
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Augmentation Types
# =============================================================================


class AugmentationType(Enum):
    """Types of data augmentation."""
    DECK_COMPOSITION = auto()
    CARD_ORDER = auto()
    OPPONENT_STRATEGY = auto()
    GAME_CONDITIONS = auto()
    ARENA_STATE = auto()
    ELIXIR_ADVANTAGE = auto()
    TIMING = auto()
    REPLAY = auto()


# =============================================================================
# Augmentation Strategies
# =============================================================================


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.

    Attributes:
        enabled: Whether augmentation is enabled.
        types: Types of augmentation to apply.
        intensity: Augmentation intensity (0.0 to 1.0).
        seed: Random seed for reproducibility.
    """
    enabled: bool = True
    types: List[AugmentationType] = field(default_factory=list)
    intensity: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.types:
            self.types = list(AugmentationType)


class DeckAugmenter:
    """Augments card deck compositions for training diversity."""

    def __init__(self, card_registry: Dict[str, Any], seed: Optional[int] = None):
        """Initialize the deck augmenter.

        Args:
            card_registry: Card definition registry.
            seed: Random seed.
        """
        self.card_registry = card_registry
        self.rng = np.random.RandomState(seed)

    def augment_deck(self, deck: List[str],
                     min_cards: int = 8,
                     max_cards: int = 10,
                     enforce_cost_range: bool = True,
                     min_cost: int = 2,
                     max_cost: int = 6) -> List[str]:
        """Augment a card deck.

        Args:
            deck: Original deck.
            min_cards: Minimum cards in augmented deck.
            max_cards: Maximum cards in augmented deck.
            enforce_cost_range: Whether to enforce cost constraints.
            min_cost: Minimum card cost.
            max_cost: Maximum card cost.

        Returns:
            Augmented deck.
        """
        if not self.rng.random() < 0.5:  # 50% chance to augment
            return deck

        augmented = deck.copy()

        # Randomly remove cards
        if len(augmented) > min_cards and self.rng.random() < 0.3:
            n_remove = self.rng.randint(1, len(augmented) - min_cards + 1)
            remove_indices = self.rng.choice(len(augmented), size=n_remove, replace=False)
            augmented = [c for i, c in enumerate(augmented) if i not in remove_indices]

        # Randomly add cards
        if len(augmented) < max_cards and self.rng.random() < 0.5:
            available_cards = list(self.card_registry.keys())
            n_add = self.rng.randint(1, max_cards - len(augmented) + 1)
            new_cards = self.rng.choice(available_cards, size=n_add, replace=False).tolist()
            augmented.extend(new_cards)

        # Enforce cost constraints
        if enforce_cost_range:
            augmented = self._enforce_cost_constraints(augmented, min_cost, max_cost)

        return augmented

    def _enforce_cost_constraints(self, deck: List[str],
                                   min_cost: int, max_cost: int) -> List[str]:
        """Enforce cost constraints on deck."""
        # This would need access to card costs
        # For now, just return the deck
        return deck

    def generate_varied_decks(self, base_deck: List[str],
                               n_decks: int = 10) -> List[List[str]]:
        """Generate multiple varied decks from a base deck.

        Args:
            base_deck: Base deck to vary.
            n_decks: Number of decks to generate.

        Returns:
            List of varied decks.
        """
        decks = [base_deck.copy()]

        for _ in range(n_decks - 1):
            deck = self.augment_deck(base_deck)
            decks.append(deck)

        return decks


class OpponentStrategyAugmenter:
    """Augments opponent strategies for training diversity."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the opponent strategy augmenter.

        Args:
            seed: Random seed.
        """
        self.rng = np.random.RandomState(seed)

    def augment_strategy(self, base_strategy: str,
                         intensity: float = 0.5) -> str:
        """Augment an opponent strategy.

        Args:
            base_strategy: Base strategy name.
            intensity: Augmentation intensity.

        Returns:
            Augmented strategy name.
        """
        strategies = ["random", "aggressive", "defensive", "balanced", "self_play"]

        # Randomly perturb strategy
        if self.rng.random() < intensity:
            idx = strategies.index(base_strategy) if base_strategy in strategies else 0
            new_idx = max(0, min(len(strategies) - 1, idx + self.rng.randint(-1, 2)))
            return strategies[new_idx]

        return base_strategy

    def generate_strategy_mix(self, n_opponents: int = 5) -> List[str]:
        """Generate a mix of opponent strategies.

        Args:
            n_opponents: Number of opponents.

        Returns:
            List of strategy names.
        """
        strategies = ["random", "aggressive", "defensive", "balanced", "self_play"]
        return [self.rng.choice(strategies) for _ in range(n_opponents)]


class GameConditionAugmenter:
    """Augments game conditions for training diversity."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the game condition augmenter.

        Args:
            seed: Random seed.
        """
        self.rng = np.random.RandomState(seed)

    def augment_conditions(self, base_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Augment game conditions.

        Args:
            base_conditions: Base game conditions.

        Returns:
            Augmented game conditions.
        """
        conditions = base_conditions.copy()

        # Augment match duration
        if "match_duration_ticks" in conditions:
            duration = conditions["match_duration_ticks"]
            noise = self.rng.normal(0, 0.1 * duration)
            conditions["match_duration_ticks"] = max(600, int(duration + noise))

        # Augment elixir regen rate
        if "elixir_regen_rate" in conditions:
            rate = conditions["elixir_regen_rate"]
            noise = self.rng.normal(0, 0.05)
            conditions["elixir_regen_rate"] = max(0.1, min(0.5, rate + noise))

        # Augment overtime
        if "overtime_ticks" in conditions:
            overtime = conditions["overtime_ticks"]
            noise = self.rng.normal(0, 0.1 * overtime)
            conditions["overtime_ticks"] = max(30, int(overtime + noise))

        return conditions

    def generate_condition_variations(self, base_conditions: Dict[str, Any],
                                       n_variations: int = 5) -> List[Dict[str, Any]]:
        """Generate variations of game conditions.

        Args:
            base_conditions: Base game conditions.
            n_variations: Number of variations to generate.

        Returns:
            List of condition dictionaries.
        """
        variations = [base_conditions.copy()]

        for _ in range(n_variations - 1):
            variation = self.augment_conditions(base_conditions)
            variations.append(variation)

        return variations


# =============================================================================
# Augmentation Pipeline
# =============================================================================


class AugmentationPipeline:
    """Pipeline for applying data augmentations.

    Chains multiple augmentation strategies together
    for comprehensive data augmentation.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initialize the augmentation pipeline.

        Args:
            config: Augmentation configuration.
        """
        self.config = config or AugmentationConfig()
        self.deck_augmenter: Optional[DeckAugmenter] = None
        self.strategy_augmenter = OpponentStrategyAugmenter(self.config.seed)
        self.condition_augmenter = GameConditionAugmenter(self.config.seed)

    def set_card_registry(self, card_registry: Dict[str, Any]) -> None:
        """Set the card registry for deck augmentation.

        Args:
            card_registry: Card definition registry.
        """
        self.deck_augmenter = DeckAugmenter(card_registry, self.config.seed)

    def augment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to data.

        Args:
            data: Input data dictionary.

        Returns:
            Augmented data dictionary.
        """
        if not self.config.enabled:
            return data

        result = data.copy()

        for aug_type in self.config.types:
            if aug_type == AugmentationType.DECK_COMPOSITION and self.deck_augmenter:
                if "deck" in result:
                    result["deck"] = self.deck_augmenter.augment_deck(
                        result["deck"],
                        intensity=self.config.intensity,
                    )

            elif aug_type == AugmentationType.OPPONENT_STRATEGY:
                if "opponent_strategy" in result:
                    result["opponent_strategy"] = self.strategy_augmenter.augment_strategy(
                        result["opponent_strategy"],
                        self.config.intensity,
                    )

            elif aug_type == AugmentationType.GAME_CONDITIONS:
                if "conditions" in result:
                    result["conditions"] = self.condition_augmenter.augment_conditions(
                        result["conditions"],
                    )

        return result

    def generate_augmented_batch(self, base_data: Dict[str, Any],
                                  n_samples: int = 10) -> List[Dict[str, Any]]:
        """Generate a batch of augmented samples.

        Args:
            base_data: Base data to augment.
            n_samples: Number of samples to generate.

        Returns:
            List of augmented data samples.
        """
        samples = [base_data.copy()]

        for _ in range(n_samples - 1):
            sample = self.augment(base_data)
            samples.append(sample)

        return samples


# =============================================================================
# Augmentation Presets
# =============================================================================


def get_light_augmentation_config() -> AugmentationConfig:
    """Get light augmentation configuration."""
    return AugmentationConfig(
        enabled=True,
        types=[
            AugmentationType.DECK_COMPOSITION,
            AugmentationType.OPPONENT_STRATEGY,
        ],
        intensity=0.3,
    )


def get_medium_augmentation_config() -> AugmentationConfig:
    """Get medium augmentation configuration."""
    return AugmentationConfig(
        enabled=True,
        types=[
            AugmentationType.DECK_COMPOSITION,
            AugmentationType.OPPONENT_STRATEGY,
            AugmentationType.GAME_CONDITIONS,
        ],
        intensity=0.5,
    )


def get_heavy_augmentation_config() -> AugmentationConfig:
    """Get heavy augmentation configuration."""
    return AugmentationConfig(
        enabled=True,
        types=list(AugmentationType),
        intensity=0.8,
    )
