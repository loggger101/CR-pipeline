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
        raise ValueError(
            f"{path} holds {genome.size} parameters but the simulator plays "
            f"{expected}-parameter policies; this looks like a Torch network "
            f"checkpoint rather than an evolved agent"
        )
    return genome.ravel()
