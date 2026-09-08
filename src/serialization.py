"""Loading helpers for the project's own checkpoint files.

PyTorch 2.6 flipped ``torch.load``'s ``weights_only`` default from False to
True. Every checkpoint this project writes holds NumPy arrays and plain
metadata alongside tensors, and the restricted unpickler rejects those:

    UnpicklingError: Unsupported global: GLOBAL numpy._core.multiarray._reconstruct

which is why loading a population checkpoint failed outright on current torch.
The files come from this pipeline, not from anywhere untrusted, so full
unpickling is the right setting -- but it belongs in one place with the reason
attached rather than being repeated at each call site.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def load_checkpoint(path: str, map_location: str = "cpu") -> Any:
    """Load a checkpoint written by this project.

    Args:
        path: File to read.
        map_location: Device to map tensors onto; CPU by default so a
            GPU-trained checkpoint opens on any machine.

    Returns:
        Whatever was saved -- usually a dict.
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def load_agent_genome(path: str) -> np.ndarray:
    """Load a single agent's evolved policy genome.

    Args:
        path: An agent checkpoint (``best_agent.pt`` or similar).

    Returns:
        The policy genome as a 1-D array.

    Raises:
        ValueError: if the file holds Torch network parameters instead. Those
            cannot be played by the simulator, and returning them would produce
            an agent that loads without complaint and behaves randomly.
    """
    from .models.policy import DEFAULT_POLICY_SPEC

    payload = load_checkpoint(path)
    if not isinstance(payload, dict):
        genome = np.asarray(payload)
    else:
        raw = payload.get("genome")
        if raw is None:
            raw = payload.get("weights")
        if raw is None:
            raise ValueError(f"{path} contains no agent parameters")
        genome = np.asarray(raw)

    expected = DEFAULT_POLICY_SPEC.num_params
    if genome.size != expected:
        # Distinguish a legitimate *older-shape* evolved checkpoint from an
        # unplayable Torch-network file. The two used to be conflated ("this
        # looks like a Torch network"), which was wrong for the 2,311-param and
        # 9,207-param genomes written by earlier versions of this project: they
        # are real evolved agents, just from before the current feature set /
        # layer widths. A genuine Torch-network payload is far larger (the CNN+LSTM
        # architectures run to millions of parameters) and/or carries no genome key.
        if genome.size < expected // 2:
            raise ValueError(
                f"{path} holds a {genome.size}-parameter evolved policy that does "
                f"not match the current default ({expected} parameters) -- it is "
                f"either from an earlier version (feature set / layer widths have "
                f"changed since 2,311-param and 9,207-param nets) or was trained "
                f"with a custom --hidden-layers shape. Re-train with the default "
                f"net, or use this file with a matching checkout."
            )
        raise ValueError(
            f"{path} holds {genome.size} parameters but the simulator plays "
            f"{expected}-parameter policies; this looks like a Torch network "
            f"checkpoint rather than an evolved agent"
        )
    return genome.ravel()
