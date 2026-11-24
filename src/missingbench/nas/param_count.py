"""Parameter accounting for the width-elastic PlainUnet supernet.

Two implementations of the same formula:

``exact_parameters``
    Integer reference.  Must agree *exactly* (not approximately) with
    ``sum(p.numel() for p in PlainUnet(features_per_stage=widths).parameters())``.

``expected_parameters``
    Differentiable version over *soft* (fractional) widths, used by the budget
    penalty during supernet training.  Substituting integer widths reproduces
    ``exact_parameters``.

Both mirror the layout of ``dynamic_network_architectures.PlainConvUNet`` as
wrapped by :class:`missingbench.models.plain_unet.PlainUnetBackbone`.  Getting a single term
wrong silently ruins the budget->parameter diagonal, so the tensor inventory
below is spelled out term by term.

Layout facts this encodes (all verified against the installed library, 0.4.3):

* Encoder stage ``s`` is ``StackedConvBlocks(n_enc, in=C[s-1], out=C[s])``: one
  conv ``C[s-1]->C[s]`` followed by ``n_enc-1`` convs ``C[s]->C[s]``.  Every
  ``ConvDropoutNormReLU`` carries a conv bias plus an InstanceNorm weight+bias,
  i.e. ``3 * out`` extra scalars.
* The decoder is built bottleneck-first.  Decoder index ``s`` upsamples from
  ``below = C[S-1-s]`` to ``skip = C[S-2-s]``.
* ``transpconvs[s]`` is a ``ConvTranspose3d`` whose weight is ``[in, out, ...]``
  -- the opposite of ``Conv3d`` -- with ``kernel = stride = 2``.
* The decoder stage conv consumes ``torch.cat((upsampled, skip), 1)``, so its
  input width is ``2 * skip``.
* ``deep_supervision=False`` still *allocates* all ``S-1`` seg layers; they are
  in ``state_dict()`` and count toward ``numel()``.  Counting only the last one
  is a real (if small) error that breaks exactness.
* ``LatentMaskConditioning`` is ``Linear(num_modals, h) -> GELU -> Linear(h, 2*C[-1])``
  with ``h = max(C[-1] // 2, num_modals * 2)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

__all__ = [
    "CONV_KERNEL_VOLUME",
    "TRANSPOSE_KERNEL_VOLUME",
    "exact_parameters",
    "expected_parameters",
    "mask_conditioning_hidden",
]

# 3x3x3 convolutions everywhere in the encoder/decoder stages ...
CONV_KERNEL_VOLUME = 27
# ... and 2x2x2 transposed convolutions (kernel == stride == 2) on the way up.
TRANSPOSE_KERNEL_VOLUME = 8

# Per ConvDropoutNormReLU, on top of the conv weight: conv bias (out) plus
# InstanceNorm affine weight and bias (2 * out).
_BIAS_AND_NORM_PER_OUT = 3


def mask_conditioning_hidden(bottleneck: int, num_modals: int = 4) -> int:
    """Hidden width of ``LatentMaskConditioning`` for a given bottleneck."""
    return max(int(bottleneck) // 2, num_modals * 2)


def _validate_widths(widths: Sequence[int] | Tensor) -> None:
    # For tensors the stage axis is the last one (leading axes are batch), so
    # len() would report the batch size instead.
    n_stages = widths.shape[-1] if isinstance(widths, Tensor) else len(widths)
    if n_stages < 4:
        raise ValueError(
            f"PlainUnetBackbone requires at least four stages; got {n_stages} widths"
        )


def exact_parameters(
    widths: Sequence[int],
    *,
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
) -> int:
    """Exact integer parameter count of ``PlainUnet(features_per_stage=widths)``."""
    _validate_widths(widths)
    c = [int(w) for w in widths]
    if any(w <= 0 for w in c):
        raise ValueError(f"widths must be positive, got {widths}")
    n_stages = len(c)
    kc = CONV_KERNEL_VOLUME
    kt = TRANSPOSE_KERNEL_VOLUME

    total = 0

    # ---- encoder -------------------------------------------------------
    for s in range(n_stages):
        fan_in = num_modals if s == 0 else c[s - 1]
        # first conv of the stage: fan_in -> c[s]
        total += c[s] * fan_in * kc + _BIAS_AND_NORM_PER_OUT * c[s]
        # remaining convs of the stage: c[s] -> c[s]
        total += (n_conv_per_stage - 1) * (
            c[s] * c[s] * kc + _BIAS_AND_NORM_PER_OUT * c[s]
        )

    # ---- decoder -------------------------------------------------------
    for s in range(n_stages - 1):
        below = c[n_stages - 1 - s]
        skip = c[n_stages - 2 - s]

        # ConvTranspose3d weight is [in, out, ...]; bias has `out` entries.
        total += below * skip * kt + skip

        # First conv consumes cat(upsampled, skip) -> 2 * skip input channels.
        total += skip * (2 * skip) * kc + _BIAS_AND_NORM_PER_OUT * skip
        total += (n_conv_per_stage_decoder - 1) * (
            skip * skip * kc + _BIAS_AND_NORM_PER_OUT * skip
        )

        # Every seg layer is allocated even with deep_supervision=False.
        total += num_cls * skip + num_cls

    # ---- mask conditioning MLP ----------------------------------------
    hidden = mask_conditioning_hidden(c[-1], num_modals)
    total += hidden * num_modals + hidden          # Linear(num_modals, hidden)
    total += (2 * c[-1]) * hidden + 2 * c[-1]      # Linear(hidden, 2 * bottleneck)

    return int(total)


def expected_parameters(
    widths: Tensor,
    *,
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
) -> Tensor:
    """Differentiable parameter count over soft widths.

    ``widths`` is ``[..., n_stages]`` and may be fractional; gradients flow back
    to the gate thresholds that produced it.  The hidden width of the mask
    conditioning MLP mirrors ``LatentMaskConditioning``'s own
    ``max(c[-1] / 2, 2 * num_modals)``; the floor is inactive for a realistically
    sized bottleneck but is what keeps this in step with ``exact_parameters`` at
    the small end.  ``clamp_min`` zeroes the gradient below the floor, which is
    correct -- the parameter count genuinely stops responding there.
    """
    _validate_widths(widths)
    if widths.shape[-1] < 4:
        raise ValueError(f"expected at least 4 stages, got shape {tuple(widths.shape)}")

    c = widths
    n_stages = c.shape[-1]
    kc = float(CONV_KERNEL_VOLUME)
    kt = float(TRANSPOSE_KERNEL_VOLUME)

    total = torch.zeros(c.shape[:-1], dtype=c.dtype, device=c.device)

    # ---- encoder -------------------------------------------------------
    for s in range(n_stages):
        fan_in = (
            torch.full_like(c[..., 0], float(num_modals)) if s == 0 else c[..., s - 1]
        )
        total = total + c[..., s] * fan_in * kc + _BIAS_AND_NORM_PER_OUT * c[..., s]
        total = total + (n_conv_per_stage - 1) * (
            c[..., s] * c[..., s] * kc + _BIAS_AND_NORM_PER_OUT * c[..., s]
        )

    # ---- decoder -------------------------------------------------------
    for s in range(n_stages - 1):
        below = c[..., n_stages - 1 - s]
        skip = c[..., n_stages - 2 - s]

        total = total + below * skip * kt + skip
        total = total + skip * (2 * skip) * kc + _BIAS_AND_NORM_PER_OUT * skip
        total = total + (n_conv_per_stage_decoder - 1) * (
            skip * skip * kc + _BIAS_AND_NORM_PER_OUT * skip
        )
        total = total + num_cls * skip + num_cls

    # ---- mask conditioning MLP ----------------------------------------
    hidden = (c[..., -1] / 2.0).clamp_min(float(num_modals * 2))
    total = total + hidden * num_modals + hidden
    total = total + (2 * c[..., -1]) * hidden + 2 * c[..., -1]

    return total
