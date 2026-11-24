from __future__ import annotations

import torch
import torch.nn.functional as F


def _threshold_to_label_one_hot(logits: torch.Tensor, num_cls: int, names: str) -> torch.Tensor:
    """Thresholds 3 sigmoid channels at ``> 0.5`` and paints labels ED (2), then NCR (1), then ET (3),
    so channel 0 wins over channel 1 and channel 1 over channel 2; returns the 4-class one-hot."""
    if logits.size(1) != 3:
        raise RuntimeError(f"expected 3 channels {names}, got {logits.size(1)}")
    on = logits > 0
    labels = torch.zeros_like(on[:, 0], dtype=torch.long)
    labels[on[:, 2]] = 2
    labels[on[:, 1]] = 1
    labels[on[:, 0]] = 3
    return F.one_hot(labels, num_classes=num_cls).permute(0, 4, 1, 2, 3).to(logits.dtype)


def regions_to_label_one_hot(region_logits: torch.Tensor, num_cls: int = 4) -> torch.Tensor:
    """Turns ``[ET, TC, WT]`` region logits into a 4-class one-hot volume whose argmax is the label map.

    Each region is thresholded at ``sigmoid > 0.5`` (legacy ``post_trans``), then nested the usual
    BraTS way: ET where ET is on, else NCR where TC is on, else ED where WT is on, else background.
    """
    return _threshold_to_label_one_hot(region_logits, num_cls, "[ET, TC, WT]")


__all__ = ["regions_to_label_one_hot"]
