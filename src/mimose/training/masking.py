from __future__ import annotations

import torch


def mae_patch_gate(
    batch_size: int,
    *,
    num_modalities: int = 4,
    patch_shape: int = 128,
    token_size: int = 16,
    mask_ratio: float = 0.875,
    mdp: int = 3,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Voxel gate for M3AE/MCPL masked-autoencoder pretraining (1 = visible, 0 = masked).

    Port of legacy ``MaskEmbeeding2`` + ``ShuffleIndex_with_MDP`` (legacy/m3ae/model/Unet.py:700-793):
    per sample, drop ``U{0..min(mdp, 3)}`` whole modalities, then keep
    ``int((1 - mask_ratio) * total_tokens)`` random ``token_size``^3 tokens among the remaining
    modalities. Upstream re-applies one batch-wide mask per sample in a loop (compounding masks when
    batch > 1); here every sample gets its own independent mask.
    """
    if patch_shape % token_size:
        raise ValueError(f"patch_shape {patch_shape} must be a multiple of token_size {token_size}")
    grid = patch_shape // token_size
    tokens_per_modality = grid**3
    total_tokens = tokens_per_modality * num_modalities
    keep_count = int((1 - mask_ratio) * total_tokens)
    max_drop = min(mdp, num_modalities - 1)

    gates = torch.zeros(batch_size, total_tokens, device=device)
    for sample in range(batch_size):
        drop_count = int(torch.randint(0, max_drop + 1, size=(1,)).item())
        dropped = torch.randperm(num_modalities)[:drop_count]
        available = torch.ones(num_modalities, tokens_per_modality, dtype=torch.bool)
        available[dropped] = False
        candidates = available.flatten().nonzero(as_tuple=True)[0]
        chosen = candidates[torch.randperm(candidates.numel())[: min(keep_count, candidates.numel())]]
        gates[sample, chosen.to(gates.device)] = 1.0

    gates = gates.view(batch_size, num_modalities, grid, grid, grid)
    for dim in (2, 3, 4):
        gates = gates.repeat_interleave(token_size, dim=dim)
    return gates


def drop_one_available(mask: torch.Tensor) -> torch.Tensor:
    """Per sample, switches off one uniformly chosen available modality of a ``[B, M]`` boolean mask.

    Samples with a single available modality are returned unchanged, so the result is always a
    non-empty subset of ``mask``.
    """
    if mask.ndim != 2:
        raise ValueError(f"expected a [B, M] modality mask, got {tuple(mask.shape)}")
    mask = mask.bool()
    available = mask.sum(dim=1)
    # Random scores on available modalities only; argmax picks one of them uniformly.
    scores = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, -1.0)
    drop = torch.zeros_like(mask)
    drop[torch.arange(mask.size(0), device=mask.device), scores.argmax(dim=1)] = True
    drop &= (available > 1).unsqueeze(1)
    return mask & ~drop


__all__ = ["drop_one_available", "mae_patch_gate"]
