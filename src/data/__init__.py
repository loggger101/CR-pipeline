"""CR-Pipeline: Data Augmentation.

Provides:
- Card deck augmentation strategies
- Opponent strategy variation
- Game condition augmentation
- Augmentation pipeline
"""

from .augmentation import (
    AugmentationConfig,
    AugmentationType,
    DeckAugmenter,
    OpponentStrategyAugmenter,
    GameConditionAugmenter,
    AugmentationPipeline,
    get_light_augmentation_config,
    get_medium_augmentation_config,
    get_heavy_augmentation_config,
)

__all__ = [
    "AugmentationConfig",
    "AugmentationType",
    "DeckAugmenter",
    "OpponentStrategyAugmenter",
    "GameConditionAugmenter",
    "AugmentationPipeline",
    "get_light_augmentation_config",
    "get_medium_augmentation_config",
    "get_heavy_augmentation_config",
]
