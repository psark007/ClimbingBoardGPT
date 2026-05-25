from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device,
    *,
    trusted: bool = False,
) -> dict[str, Any]:
    """Load a PyTorch checkpoint, preferring safer weights-only loading.

    Set ``trusted=True`` only for checkpoints produced by this project or an
    otherwise trusted source. Older PyTorch versions do not support
    ``weights_only``; those fall back to the legacy loader for compatibility.
    """
    checkpoint_path = Path(checkpoint_path)

    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        if not trusted:
            raise RuntimeError(
                "Could not load checkpoint with weights_only=True. "
                "Only retry with trusted=True for checkpoints from a trusted source."
            ) from exc
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
