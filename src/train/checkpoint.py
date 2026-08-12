"""Checkpoint management for saving and loading training state.

Handles:
- Population checkpoints (weights, fitness history)
- Best agent snapshots
- Training config serialization
- Checkpoint lifecycle (save, load, cleanup)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint save/load operations.

    Organizes checkpoints by generation with:
    - Population weights
    - Fitness history
    - Training metadata
    - Best agent snapshot
    """

    def __init__(self, runs_dir: str = "runs", max_checkpoints: int = 50):
        """Initialize checkpoint manager.

        Args:
            runs_dir: Directory for training outputs.
            max_checkpoints: Maximum checkpoints to retain.
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints

        # Create subdirectories
        (self.runs_dir / "best").mkdir(parents=True, exist_ok=True)
        (self.runs_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        generation: int,
        population_weights: List[np.ndarray],
        fitness_history: Dict[str, List[float]],
        metadata: Dict,
    ) -> str:
        """Save a generation checkpoint.

        Args:
            generation: Generation number.
            population_weights: List of weight arrays.
            fitness_history: Fitness history dictionary.
            metadata: Additional training metadata.

        Returns:
            Path to the saved checkpoint.
        """
        gen_dir = self.runs_dir / "checkpoints" / f"gen_{generation:04d}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save population weights
        weights_path = gen_dir / "population.pt"
        torch.save({
            "weights": [w.tolist() for w in population_weights],
            "generation": generation,
        }, weights_path)

        # Save fitness history
        history_path = gen_dir / "fitness_history.json"
        with open(history_path, "w") as f:
            json.dump(fitness_history, f, indent=2)

        # Save metadata
        meta_path = gen_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint gen {generation} to {gen_dir}")
        return str(gen_dir)

    def save_best_agent(
        self,
        weights: np.ndarray,
        fitness: float,
        generation: int,
    ) -> str:
        """Save the best agent snapshot.

        Args:
            weights: Best agent weights.
            fitness: Best fitness score.
            generation: Generation at which this was achieved.

        Returns:
            Path to the saved best agent.
        """
        best_dir = self.runs_dir / "best"

        # Save weights
        weights_path = best_dir / "weights.pt"
        torch.save({
            "weights": weights.tolist(),
            "fitness": fitness,
            "generation": generation,
            "timestamp": str(os.path.getmtime(str(best_dir / "weights.pt")))
            if (best_dir / "weights.pt").exists() else 0,
        }, weights_path)

        # Save metadata
        meta_path = best_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump({
                "fitness": fitness,
                "generation": generation,
                "timestamp": str(os.path.getmtime(str(weights_path))),
            }, f, indent=2)

        logger.info(f"Saved best agent (fitness={fitness:.3f}, gen={generation})")
        return str(best_dir)

    def load_checkpoint(self, gen: int) -> Dict:
        """Load a generation checkpoint.

        Args:
            gen: Generation number to load.

        Returns:
            Dictionary with weights, history, and metadata.
        """
        gen_dir = self.runs_dir / "checkpoints" / f"gen_{gen:04d}"

        if not gen_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {gen_dir}")

        # Load weights
        weights_path = gen_dir / "population.pt"
        checkpoint = torch.load(weights_path)
        weights = [np.array(w) for w in checkpoint["weights"]]

        # Load history
        history_path = gen_dir / "fitness_history.json"
        with open(history_path) as f:
            history = json.load(f)

        # Load metadata
        meta_path = gen_dir / "metadata.json"
        with open(meta_path) as f:
            metadata = json.load(f)

        return {
            "weights": weights,
            "fitness_history": history,
            "metadata": metadata,
            "generation": checkpoint.get("generation", gen),
        }

    def load_best_agent(self) -> Optional[Dict]:
        """Load the best agent snapshot.

        Returns:
            Dictionary with weights and metadata, or None if not found.
        """
        best_dir = self.runs_dir / "best"
        weights_path = best_dir / "weights.pt"

        if not weights_path.exists():
            return None

        checkpoint = torch.load(weights_path)
        return {
            "weights": np.array(checkpoint["weights"]),
            "fitness": checkpoint["fitness"],
            "generation": checkpoint["generation"],
        }

    def list_checkpoints(self) -> List[int]:
        """List available checkpoint generations.

        Returns:
            Sorted list of checkpoint generation numbers.
        """
        checkpoints_dir = self.runs_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return []

        generations = []
        for d in checkpoints_dir.iterdir():
            if d.is_dir() and d.name.startswith("gen_"):
                try:
                    gen = int(d.name.split("_")[1])
                    generations.append(gen)
                except (ValueError, IndexError):
                    pass

        return sorted(generations)

    def cleanup_old_checkpoints(self) -> int:
        """Remove old checkpoints beyond max_checkpoints.

        Returns:
            Number of checkpoints removed.
        """
        generations = self.list_checkpoints()
        removed = 0

        if len(generations) > self.max_checkpoints:
            to_remove = generations[:len(generations) - self.max_checkpoints]
            for gen in to_remove:
                gen_dir = self.runs_dir / "checkpoints" / f"gen_{gen:04d}"
                import shutil
                if gen_dir.exists():
                    shutil.rmtree(gen_dir)
                    removed += 1
                    logger.info(f"Removed checkpoint: gen_{gen:04d}")

        return removed

    def get_checkpoint_info(self, gen: int) -> Optional[Dict]:
        """Get information about a specific checkpoint.

        Args:
            gen: Generation number.

        Returns:
            Dictionary with checkpoint info, or None if not found.
        """
        gen_dir = self.runs_dir / "checkpoints" / f"gen_{gen:04d}"

        if not gen_dir.exists():
            return None

        meta_path = gen_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)

        return {
            "generation": gen,
            "exists": True,
        }
