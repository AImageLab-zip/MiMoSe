from __future__ import annotations

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel

basic_dims = 16
num_modals = 4
input_patch_size = 80
DATASET_MODALITY_ORDER = (2, 0, 1, 3)

A2FSegOutput = torch.Tensor | tuple[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]


def normalization(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(planes)
    if norm == "gn":
        return nn.GroupNorm(4, planes)
    if norm == "in":
        return nn.InstanceNorm3d(planes)
    raise ValueError(f"normalization type {norm} is not supported")


class GeneralConv3d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.01,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            padding_mode=pad_type,
            bias=True,
        )
        self.norm = normalization(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(
                negative_slope=relufactor,
                inplace=True,
            )
        else:
            raise ValueError(f"activation type {act_type} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


def masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average per-modality feature maps over the present-modality axis.

    ``features`` is stacked as ``[B, M, C, H, W, D]`` and ``mask`` is
    ``[B, M]``. This mirrors legacy A2FSeg's fusion-decoder skip aggregation
    (``torch.mean(torch.stack(present_modality_features, 0), 0)``), except
    legacy always assumes batch size 1 and physically never runs the encoder
    for a dropped modality, whereas here every sample in the batch may drop a
    different subset, so the mean is computed per-sample over the present
    subset instead.
    """
    weights = mask.to(features.dtype).view(mask.size(0), mask.size(1), 1, 1, 1, 1)
    summed = (features * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return summed / count


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = GeneralConv3d(1, basic_dims, pad_type="reflect")
        self.e1_c2 = GeneralConv3d(basic_dims, basic_dims, pad_type="reflect")
        self.e1_c3 = GeneralConv3d(basic_dims, basic_dims, pad_type="reflect")

        self.e2_c1 = GeneralConv3d(basic_dims, basic_dims * 2, stride=2, pad_type="reflect")
        self.e2_c2 = GeneralConv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = GeneralConv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")

        self.e3_c1 = GeneralConv3d(basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect")
        self.e3_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = GeneralConv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")

        self.e4_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect")
        self.e4_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = GeneralConv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        return x1, x2, x3, x4


class DecoderSep(nn.Module):
    """Per-modality decoder, weight-shared across the 4 modality branches.

    Mirrors legacy A2FSeg's independent modality-specific encoder-decoder
    (one of the 4 ``Generic_UNet`` instances in ``modality_specific_models``),
    simplified to a single shared decoder applied to each modality's encoder
    features in turn -- the same simplification already used by RFNet/IMFuse
    in this codebase. Returns every intermediate per-stage decoded feature
    (not just the final one), since legacy's fusion decoder (see
    ``DecoderFuse``) consumes each modality's own *decoded* stage outputs as
    its skip-fusion inputs at every level except the bottleneck -- along with
    the per-modality softmax prediction (the auxiliary ``sep_preds`` branch
    for the IMFuse-style training loss).
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = GeneralConv3d(
            basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect"
        )

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(
            basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect"
        )

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = GeneralConv3d(
            basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect"
        )

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        up4 = self.d3_c1(self.d3(x4))
        stage1 = self.d3_out(self.d3_c2(torch.cat((up4, x3), dim=1)))

        up3 = self.d2_c1(self.d2(stage1))
        stage2 = self.d2_out(self.d2_c2(torch.cat((up3, x2), dim=1)))

        up2 = self.d1_c1(self.d1(stage2))
        stage3 = self.d1_out(self.d1_c2(torch.cat((up2, x1), dim=1)))

        pred = self.softmax(self.seg_layer(stage3))
        return stage1, stage2, stage3, pred


class DecoderFuse(nn.Module):
    """Adaptive multi-modal fusion decoder (the "A2F" module).

    Legacy's ``Generic_UNet_Decoder`` only averages *raw encoder* features at
    the coarsest (bottleneck) level; at every shallower level it averages
    each present modality's own *already-decoded* stage output (i.e. that
    modality's own upsample -> skip-concat -> conv result from its
    independent decoder), then re-decodes that per-level ensemble through
    its own upsample/concat/conv stages. This mirrors that exactly: the
    bottleneck skip (``x4``) comes straight from the encoders, but every
    other skip (``stage1_feats``/``stage2_feats``/``stage3_feats``) is the
    corresponding intermediate feature already produced by each modality's
    own ``DecoderSep`` pass, not a raw encoder feature. Averaging still uses
    a per-sample masked mean (legacy's ``torch.mean(torch.stack(...), 0)``
    generalized to a batch where different samples may drop different
    modalities, unlike legacy's batch-size-1 assumption).

    On top of that, for every present modality a small "modality-aware"
    conv stack takes the concatenation of that modality's own final decoded
    feature (``stage3_feats``, i.e. ``DecoderSep``'s finest-resolution
    output) and this fused feature, producing a per-voxel, per-channel
    gating map. The gating maps are stacked over the modality axis and
    passed through a softmax restricted to the present modalities (legacy
    drops absent-modality branches from the stack entirely before the
    softmax; here they are masked to -inf instead, which is equivalent after
    softmax). The softmax-weighted sum of per-modality features is
    concatenated with the fused feature and projected to the segmentation
    output -- this is legacy's single-scale (final-resolution only) adaptive
    fusion; the multi-resolution deep-supervision heads used by legacy's
    nnU-Net trainer are dropped, matching how deep supervision is already
    omitted for the other ported models in this codebase.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d3_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = GeneralConv3d(
            basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect"
        )

        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(
            basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect"
        )

        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = GeneralConv3d(
            basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect"
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        # Legacy's modality-aware gate is conv->norm->lrelu(0.01) followed by
        # a *second* conv with no normalization before its LeakyReLU (default
        # slope 0.01) -- unlike GeneralConv3d's other uses in this file, the
        # second layer here is built from a raw Conv3d so it doesn't pick up
        # an extra InstanceNorm3d that legacy never applies.
        self.modality_aware = nn.ModuleList(
            [
                nn.Sequential(
                    GeneralConv3d(basic_dims * 2, basic_dims, k_size=3, padding=1, pad_type="reflect"),
                    nn.Conv3d(basic_dims, basic_dims, kernel_size=3, padding=1, bias=True),
                    nn.LeakyReLU(),
                )
                for _ in range(num_modals)
            ]
        )
        self.lastconv = GeneralConv3d(basic_dims * 2, basic_dims, k_size=3, padding=1, pad_type="reflect")
        self.output = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x4: torch.Tensor,
        stage1_feats: list[torch.Tensor],
        stage2_feats: list[torch.Tensor],
        stage3_feats: list[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        fused_x4 = masked_mean(x4, mask)
        de_x4 = self.d3_c1(self.up2(fused_x4))

        fused_stage1 = masked_mean(torch.stack(stage1_feats, dim=1), mask)
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x4, fused_stage1), dim=1)))
        de_x3 = self.d2_c1(self.up2(de_x3))

        fused_stage2 = masked_mean(torch.stack(stage2_feats, dim=1), mask)
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x3, fused_stage2), dim=1)))
        de_x2 = self.d1_c1(self.up2(de_x2))

        fused_stage3 = masked_mean(torch.stack(stage3_feats, dim=1), mask)
        fused_feat = self.d1_out(self.d1_c2(torch.cat((de_x2, fused_stage3), dim=1)))

        attention_scores = [
            self.modality_aware[index](torch.cat((feat, fused_feat), dim=1))
            for index, feat in enumerate(stage3_feats)
        ]
        attention_stack = torch.stack(attention_scores, dim=1)
        present = mask.view(mask.size(0), mask.size(1), 1, 1, 1, 1)
        attention_stack = attention_stack.masked_fill(~present, torch.finfo(attention_stack.dtype).min)
        attention_weights = torch.softmax(attention_stack, dim=1)

        stacked_feats = torch.stack(stage3_feats, dim=1)
        weighted_sum = (stacked_feats * attention_weights).sum(dim=1)

        combined = self.lastconv(torch.cat((weighted_sum, fused_feat), dim=1))
        return self.softmax(self.output(combined))


class A2FSeg(AbstractModel):
    """Faithful port of A2FSeg (MICCAI 2023) to the MissingBench IMFuse-style contract.

    Legacy runs 4 fully independent per-modality nnU-Net encoder-decoders
    plus a separate fusion decoder that averages present-modality features
    at every skip level and adaptively (softmax-gated) recombines each
    modality's own decoder output. This port keeps that two-decoder
    structure (see ``DecoderFuse``) but, like the other ported models here,
    weight-shares the per-modality decoder (``DecoderSep``) instead of using
    4 independent decoders, uses a smaller 4-level encoder at the codebase's
    80^3 patch size instead of legacy's ~5-level encoder at 128^3, replaces
    the sigmoid/BCE 3-region BraTS output with a softmax num_cls-channel
    head, and drops the multi-resolution deep-supervision heads -- all
    consistent with how ShaSpec/LCKD/RFNet/IMFuse were already adapted in
    this codebase. A missing modality's raw input is zero-filled rather than
    the encoder being skipped entirely (legacy assumes batch size 1 and
    literally never runs a dropped modality's branch); the softmax fusion
    gate and skip-mean are masked so a missing modality still contributes
    nothing to the fused output, which is the behavior that matters.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.decoder_sep = DecoderSep(num_cls=num_cls)
        self.decoder_fuse = DecoderFuse(num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> A2FSegOutput:
        x, mask = self._remap_input_order(x, mask)
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        stage1_feats = []
        stage2_feats = []
        stage3_feats = []
        sep_preds = []
        for feats in (
            (flair_x1, flair_x2, flair_x3, flair_x4),
            (t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4),
            (t1_x1, t1_x2, t1_x3, t1_x4),
            (t2_x1, t2_x2, t2_x3, t2_x4),
        ):
            stage1, stage2, stage3, pred = self.decoder_sep(*feats)
            stage1_feats.append(stage1)
            stage2_feats.append(stage2)
            stage3_feats.append(stage3)
            sep_preds.append(pred)

        fuse_pred = self.decoder_fuse(x4, stage1_feats, stage2_feats, stage3_feats, mask)

        if self.is_training:
            return fuse_pred, tuple(sep_preds), ()
        return fuse_pred

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.decoder_fuse.output.out_channels,
            height,
            width,
            depth,
            device=images.device,
        )
        weight = torch.zeros(
            images.size(0),
            1,
            height,
            width,
            depth,
            device=images.device,
        )

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ]
                    patch_pred = self(patch, mask)
                    prediction[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + input_patch_size,
                        w : w + input_patch_size,
                        d : d + input_patch_size,
                    ] += 1
        return prediction / weight

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                "Model prediction expects spatial dimensions to be at least "
                f"{input_patch_size}, got {size}"
            )
        if size == input_patch_size:
            return [0]

        stride = input_patch_size // 2
        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(
                f"A2FSeg expects {num_modals} input modalities, got {images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(
                f"A2FSeg expects mask shape [B, {num_modals}], got {tuple(mask.shape)}"
            )

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = A2FSeg


__all__ = ["A2FSeg", "Model"]
