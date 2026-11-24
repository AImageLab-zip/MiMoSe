"""Extract a standalone ``PlainUnet`` from the supernet, with inherited weights.

Most tensors slice as a plain prefix, because the ordered gate parameterization
guarantees the active channel set is ``arange(c)``.  **Two do not**, and getting
either wrong produces a model that loads cleanly, runs, and is silently wrong:

``decoder.stages[s].convs[0].conv``
    Its input is ``torch.cat((upsampled, skip), 1)``, so the input axis has
    width ``2 * C_max``.  The live channels are *not* ``[0 : 2c]`` -- they are
    ``[0 : c]`` from the upsampled half and ``[C_max : C_max + c]`` from the skip
    half.  A contiguous slice silently reinterprets part of the upsampled
    features as skip features.

``mask_conditioning.mlp[2]``
    Its output is split by ``.chunk(2, dim=1)`` into ``scale`` and ``bias``, so
    the same two-block layout applies to the *output* axis.

:func:`concat_indices` builds the correct index for both; replacing either with a
contiguous slice silently breaks the extracted subnetwork.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from missingbench.nas.budget import budget_to_z

if TYPE_CHECKING:  # pragma: no cover - typing only
    from missingbench.models.matunet import MatUnet
    from missingbench.models.plain_unet import PlainUnet

__all__ = ["active_indices", "concat_indices", "extract_subnet"]


def active_indices(gates, z: Tensor) -> list[Tensor]:
    """Per-stage indices of the channels active at ``z`` (always a prefix)."""
    widths = gates.hard_widths(z).reshape(-1)
    # On ``widths.device``, not the default one: these index the supernet's own
    # weights via ``index_select``, which requires matching devices.
    return [torch.arange(int(w), device=widths.device) for w in widths]


def concat_indices(idx: Tensor, max_channels: int) -> Tensor:
    """Index a concatenated ``[first_half | second_half]`` axis.

    ``max_channels`` is the width of *one* half.  The result selects ``idx`` from
    each half, which is what a ``torch.cat`` of two gated tensors leaves live.
    """
    return torch.cat([idx, max_channels + idx])


def _copy_conv(dst_block, src_block, out_idx: Tensor, in_idx: Tensor) -> None:
    """Copy one ``ConvDropoutNormReLU`` (conv weight/bias + norm affine)."""
    dst_block.conv.weight.copy_(
        src_block.conv.weight.index_select(0, out_idx).index_select(1, in_idx)
    )
    if dst_block.conv.bias is not None:
        dst_block.conv.bias.copy_(src_block.conv.bias.index_select(0, out_idx))

    dst_norm = getattr(dst_block, "norm", None)
    src_norm = getattr(src_block, "norm", None)
    if dst_norm is not None and src_norm is not None:
        if getattr(dst_norm, "weight", None) is not None:
            dst_norm.weight.copy_(src_norm.weight.index_select(0, out_idx))
        if getattr(dst_norm, "bias", None) is not None:
            dst_norm.bias.copy_(src_norm.bias.index_select(0, out_idx))


@torch.no_grad()
def extract_subnet(supernet: MatUnet, budget: float) -> PlainUnet:
    """Build the standalone ``PlainUnet`` selected by ``budget``.

    The returned model reproduces the supernet's gated forward for the same
    budget to floating-point roundoff.
    """
    from missingbench.models.plain_unet import PlainUnet

    gates = supernet.gates
    device = next(supernet.parameters()).device
    z = budget_to_z(budget, gates.budget_min, gates.budget_max)
    idx = active_indices(gates, z)
    widths = tuple(int(i.numel()) for i in idx)

    subnet = PlainUnet(
        num_cls=supernet.num_cls,
        num_modals=supernet.num_modals,
        input_shape=supernet.supernet.input_shape,
        features_per_stage=widths,
        n_conv_per_stage=supernet.n_conv_per_stage,
        n_conv_per_stage_decoder=supernet.n_conv_per_stage_decoder,
        overlap=supernet.supernet.overlap,
    )
    subnet = subnet.to(device)

    src = supernet.supernet.backbone
    dst = subnet.backbone
    n_stages = len(widths)
    modality_idx = torch.arange(supernet.num_modals, device=device)

    # ---- encoder --------------------------------------------------------
    for stage in range(n_stages):
        out_idx = idx[stage]
        # Stage 0 reads the raw modalities, which are never gated.
        fan_in_idx = modality_idx if stage == 0 else idx[stage - 1]

        src_blocks = src.encoder.stages[stage][0].convs
        dst_blocks = dst.encoder.stages[stage][0].convs
        for block_index, (dst_block, src_block) in enumerate(
            zip(dst_blocks, src_blocks)
        ):
            in_idx = fan_in_idx if block_index == 0 else out_idx
            _copy_conv(dst_block, src_block, out_idx, in_idx)

    # ---- decoder --------------------------------------------------------
    for s in range(n_stages - 1):
        level = n_stages - 2 - s          # resolution level this stage outputs
        below_idx = idx[level + 1]
        skip_idx = idx[level]

        # ConvTranspose3d weight is [in, out, ...] -- the transpose of Conv3d.
        dst.decoder.transpconvs[s].weight.copy_(
            src.decoder.transpconvs[s]
            .weight.index_select(0, below_idx)
            .index_select(1, skip_idx)
        )
        if dst.decoder.transpconvs[s].bias is not None:
            dst.decoder.transpconvs[s].bias.copy_(
                src.decoder.transpconvs[s].bias.index_select(0, skip_idx)
            )

        src_blocks = src.decoder.stages[s].convs
        dst_blocks = dst.decoder.stages[s].convs
        for block_index, (dst_block, src_block) in enumerate(
            zip(dst_blocks, src_blocks)
        ):
            if block_index == 0:
                # THE gather: input axis is cat(upsampled, skip), so the live
                # channels live in two blocks, not one contiguous run.
                in_idx = concat_indices(skip_idx, supernet.max_features[level])
            else:
                in_idx = skip_idx
            _copy_conv(dst_block, src_block, skip_idx, in_idx)

        # Seg layers: output axis is the class axis and is never gated.
        class_idx = torch.arange(supernet.num_cls, device=device)
        dst.decoder.seg_layers[s].weight.copy_(
            src.decoder.seg_layers[s]
            .weight.index_select(0, class_idx)
            .index_select(1, skip_idx)
        )
        dst.decoder.seg_layers[s].bias.copy_(src.decoder.seg_layers[s].bias)

    # ---- mask conditioning ---------------------------------------------
    src_mlp = supernet.supernet.mask_conditioning.mlp
    dst_mlp = subnet.mask_conditioning.mlp
    bottleneck_idx = idx[-1]
    hidden = dst_mlp[0].out_features
    hidden_idx = torch.arange(hidden, device=device)

    dst_mlp[0].weight.copy_(src_mlp[0].weight.index_select(0, hidden_idx))
    dst_mlp[0].bias.copy_(src_mlp[0].bias.index_select(0, hidden_idx))

    # Second gather: the output is chunk(2)'d into scale|bias.
    out_idx = concat_indices(bottleneck_idx, supernet.max_features[-1])
    dst_mlp[2].weight.copy_(
        src_mlp[2].weight.index_select(0, out_idx).index_select(1, hidden_idx)
    )
    dst_mlp[2].bias.copy_(src_mlp[2].bias.index_select(0, out_idx))

    subnet._missingbench_model_kwargs = {
        "num_cls": supernet.num_cls,
        "num_modals": supernet.num_modals,
        "input_shape": tuple(int(d) for d in supernet.supernet.input_shape),
        "features_per_stage": widths,
        "n_conv_per_stage": supernet.n_conv_per_stage,
        "n_conv_per_stage_decoder": supernet.n_conv_per_stage_decoder,
        "overlap": supernet.supernet.overlap,
    }
    subnet._missingbench_model_name = "PlainUnet"
    return subnet
