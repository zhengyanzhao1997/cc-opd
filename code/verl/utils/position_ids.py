"""Lightweight position-id helpers.

Keep this module free of Hugging Face model imports. It is imported by Ray
control actors and dataset/rollout code paths that may not own a GPU.
"""

import torch


def compute_position_id_with_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
