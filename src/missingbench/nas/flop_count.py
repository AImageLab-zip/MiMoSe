"""Multiply-accumulate accounting for the width-elastic PlainUnet supernet.

The FLOPs twin of :mod:`missingbench.nas.param_count`, and deliberately the same shape:

``exact_macs``
    Integer reference for a concrete architecture.

``expected_macs``
    Differentiable version over *soft* (fractional) widths, used by the budget
    penalty when ``budget_metric="flops"``.  Substituting integer widths
    reproduces ``exact_macs``.

Why this exists at all: parameters and MACs live in *different halves* of a
U-Net.  Parameters concentrate in the deep, wide, low-resolution stages, where a
channel costs ``C_in * C_out`` and almost no arithmetic; MACs concentrate in the
shallow, high-resolution stages, where the same channel is multiplied by a
spatial volume 8x larger per level up.  A parameter budget therefore buys almost
no speed -- measured on a trained MatUnet supernet, a 16.6x parameter range
spans only 1.12 .. 1.70 TMACs and 16% wall-clock.  Budgeting MACs instead forces
the gates to cut where the time actually goes.

The term inventory mirrors ``param_count`` exactly, with every conv weighted by
the spatial volume of its *output*.  Two places where that differs from a naive
reading of the parameter formula, and both matter:

* ``strides=[1] + [2] * (n_stages - 1)`` (see ``PlainUnetBackbone.__init__``), so
  encoder stage ``s`` produces its output at ``R / 2**s`` per axis, i.e. a volume
  ``V[s] = V[0] / 8**s``.  The stride lives on the *first* conv of the stage, but
  every conv in the stage writes at the stage's output resolution.
* ``deep_supervision=False`` allocates all ``S-1`` seg layers -- they count
  toward ``numel()``, which is why ``exact_parameters`` includes them -- but only
  the last one is *evaluated*.  Counting all of them here overstates MACs.

Validated against measured ``flops_gpu.csv`` for six trained architectures
spanning 1.4M .. 23.7M parameters: agreement is within 0.07%, the residual being
the elementwise norm/activation work the profiler counts and this conv-only
model does not.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from missingbench.nas.param_count import CONV_KERNEL_VOLUME, TRANSPOSE_KERNEL_VOLUME

__all__ = [
    "exact_macs",
    "expected_macs",
    "stage_volumes",
]


def stage_volumes(input_shape: Sequence[int], n_stages: int) -> list[int]:
    """Output voxel count of each encoder stage, ``V[s] = prod(shape) / 8**s``.

    Computed by halving each axis rather than dividing the product, so a
    non-cubic input shape (or one that stops being divisible partway down)
    matches what the network actually allocates.
    """
    shape = [int(d) for d in input_shape]
    if len(shape) != 3 or any(d <= 0 for d in shape):
        raise ValueError(f"input_shape must be three positive dims, got {input_shape}")

    volumes = []
    for stage in range(n_stages):
        dims = [max(d >> stage, 1) for d in shape]
        volumes.append(dims[0] * dims[1] * dims[2])
    return volumes


def _validate(widths: Sequence[int] | Tensor) -> None:
    n_stages = widths.shape[-1] if isinstance(widths, Tensor) else len(widths)
    if n_stages < 4:
        raise ValueError(
            f"PlainUnetBackbone requires at least four stages; got {n_stages} widths"
        )


def exact_macs(
    widths: Sequence[int],
    *,
    input_shape: Sequence[int] = (128, 128, 128),
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
) -> int:
    """Exact MAC count of one forward pass of ``PlainUnet(features_per_stage=widths)``."""
    _validate(widths)
    c = [int(w) for w in widths]
    if any(w <= 0 for w in c):
        raise ValueError(f"widths must be positive, got {widths}")

    n_stages = len(c)
    v = stage_volumes(input_shape, n_stages)
    kc = CONV_KERNEL_VOLUME
    kt = TRANSPOSE_KERNEL_VOLUME

    total = 0

    # ---- encoder -------------------------------------------------------
    for s in range(n_stages):
        fan_in = num_modals if s == 0 else c[s - 1]
        total += c[s] * fan_in * kc * v[s]
        total += (n_conv_per_stage - 1) * (c[s] * c[s] * kc * v[s])

    # ---- decoder -------------------------------------------------------
    for s in range(n_stages - 1):
        below = c[n_stages - 1 - s]
        skip = c[n_stages - 2 - s]
        level = n_stages - 2 - s

        # ConvTranspose3d with kernel == stride == 2: k**3 MACs per *input*
        # voxel, and the input sits one level below the output.
        total += below * skip * kt * v[level + 1]

        # First decoder conv consumes cat(upsampled, skip) -> 2 * skip inputs.
        total += skip * (2 * skip) * kc * v[level]
        total += (n_conv_per_stage_decoder - 1) * (skip * skip * kc * v[level])

    # Only the final seg layer is evaluated (deep_supervision=False); it sits at
    # the shallowest decoder level, i.e. full resolution.
    total += num_cls * c[0] * v[0]

    return int(total)


def expected_macs(
    widths: Tensor,
    *,
    input_shape: Sequence[int] = (128, 128, 128),
    num_modals: int = 4,
    num_cls: int = 4,
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
) -> Tensor:
    """Differentiable MAC count over soft widths, shaped ``[...]``.

    ``widths`` is ``[..., n_stages]`` and may be fractional; gradients flow back
    to the gate thresholds that produced it.  Unlike ``expected_parameters``
    there is no floor term to mirror -- the mask-conditioning MLP operates on a
    ``[B, num_modals]`` vector and its arithmetic is negligible next to a single
    3D convolution, so it is omitted here rather than modelled.
    """
    _validate(widths)

    c = widths
    n_stages = c.shape[-1]
    v = stage_volumes(input_shape, n_stages)
    kc = float(CONV_KERNEL_VOLUME)
    kt = float(TRANSPOSE_KERNEL_VOLUME)

    total = torch.zeros(c.shape[:-1], dtype=c.dtype, device=c.device)

    # ---- encoder -------------------------------------------------------
    for s in range(n_stages):
        fan_in = (
            torch.full_like(c[..., 0], float(num_modals)) if s == 0 else c[..., s - 1]
        )
        total = total + c[..., s] * fan_in * kc * v[s]
        total = total + (n_conv_per_stage - 1) * (c[..., s] * c[..., s] * kc * v[s])

    # ---- decoder -------------------------------------------------------
    for s in range(n_stages - 1):
        below = c[..., n_stages - 1 - s]
        skip = c[..., n_stages - 2 - s]
        level = n_stages - 2 - s

        total = total + below * skip * kt * v[level + 1]
        total = total + skip * (2 * skip) * kc * v[level]
        total = total + (n_conv_per_stage_decoder - 1) * (skip * skip * kc * v[level])

    total = total + num_cls * c[..., 0] * v[0]

    return total
