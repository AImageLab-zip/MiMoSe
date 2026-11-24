from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

num_modals = 4
input_patch_size = 128

# Legacy LCKD (folder name; the live model class is ``DualNet``) hardcodes
# these architecture hyperparameters via its own CLI defaults (norm_cfg='IN',
# activation_cfg='LeakyReLU', weight_std=True) and never actually exercises
# any other combination in its own train.py entrypoint, so this port fixes
# them as constants instead of re-exposing a configurable-but-never-varied
# surface.
WEIGHT_STD = True

# External pipeline uses [t1c, t1n, t2f, t2w]; legacy LCKD's own
# BraTSDataSet.py stacks modalities as [flair, t1, t1ce, t2] (confirmed by
# reading its `np.stack([flair, t1, t1ce, t2], axis=0)` calls and the
# flair/t1ce indexing used inside DualNet.forward), which is yet another
# t1/t1ce ordering variant distinct from the DCSeg/RFNet family's
# [flair, t1ce, t1, t2].
DATASET_MODALITY_ORDER = (2, 1, 0, 3)


class Conv3dWD(nn.Conv3d):
    """Weight-standardized 3D convolution (legacy ``Conv3d_wd``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2, 3, 4), keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(
    in_planes: int,
    out_planes: int,
    kernel_size,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
    groups: int = 1,
    bias: bool = False,
) -> nn.Conv3d:
    if WEIGHT_STD:
        return Conv3dWD(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        dilation=dilation, groups=groups, bias=bias,
    )


class DegenerateInstanceNorm3d(nn.InstanceNorm3d):
    """InstanceNorm3d that tolerates a 1x1x1 spatial input (ASPP's global
    pooling branch), matching legacy's workaround for PyTorch's strict
    single-element check."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() > x.size(0) * x.size(1):
            return super().forward(x)
        mean = x.mean([2, 3, 4], keepdim=True)
        out = (x - mean) / math.sqrt(self.eps)
        if self.affine:
            out = out * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return out


def norm_layer(channels: int) -> nn.Module:
    return DegenerateInstanceNorm3d(channels, affine=True)


def activation_layer(inplace: bool = True) -> nn.Module:
    return nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)


class ASPP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(conv3x3x3(dim_in, dim_out, kernel_size=1), norm_layer(dim_out), activation_layer())
        self.branch2 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=2, dilation=2), norm_layer(dim_out), activation_layer()
        )
        self.branch3 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=4, dilation=4), norm_layer(dim_out), activation_layer()
        )
        self.branch4 = nn.Sequential(
            conv3x3x3(dim_in, dim_out, kernel_size=3, stride=1, padding=8, dilation=8), norm_layer(dim_out), activation_layer()
        )
        self.branch5_conv = conv3x3x3(dim_in, dim_out, kernel_size=1)
        self.branch5_norm = norm_layer(dim_out)
        self.branch5_nonlin = activation_layer()
        self.conv_cat = nn.Sequential(conv3x3x3(dim_out * 5, dim_out, kernel_size=1), norm_layer(dim_out), activation_layer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, [2, 3, 4], True)
        global_feature = self.branch5_nonlin(self.branch5_norm(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (d, h, w), mode="trilinear", align_corners=True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        return self.conv_cat(feature_cat)


class Bottleneck(nn.Module):
    """Legacy ``Res50.Bottleneck`` (ResNet50-style, expansion=4)."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride=(1, 1, 1), downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=1)
        self.norm1 = norm_layer(planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.norm2 = norm_layer(planes)
        self.conv3 = conv3x3x3(planes, planes * self.expansion, kernel_size=1)
        self.norm3 = norm_layer(planes * self.expansion)
        self.nonlin = activation_layer()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.nonlin(self.norm1(self.conv1(x)))
        out = self.nonlin(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.nonlin(out + residual)


class BasicBlock(nn.Module):
    """Legacy ``Res50.BasicBlock`` (used by the decoder refinement stages)."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.norm1 = norm_layer(planes)
        self.conv2 = conv3x3x3(inplanes, planes, kernel_size=3, stride=(1, 1, 1), padding=1)
        self.norm2 = norm_layer(planes)
        self.nonlin = activation_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.nonlin(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.nonlin(out + residual)


def _make_resnet_layer(inplanes: int, planes: int, blocks: int, stride) -> tuple[nn.Sequential, int]:
    downsample = nn.Sequential(
        conv3x3x3(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride),
        norm_layer(planes * Bottleneck.expansion),
    )
    layers = [Bottleneck(inplanes, planes, stride=stride, downsample=downsample)]
    inplanes = planes * Bottleneck.expansion
    for _ in range(1, blocks):
        layers.append(Bottleneck(inplanes, planes))
    return nn.Sequential(*layers), inplanes


class _ResNet50Backbone(nn.Module):
    """Legacy ``Res50.ResNet(depth=50, ...)``, specialized to the single
    (in_channels=1) input configuration DualNet always uses (one channel per
    modality, batch-folded)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(1, 64, kernel_size=7, stride=(1, 2, 2), padding=3)
        self.norm1 = norm_layer(64)
        self.nonlin = activation_layer()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        inplanes = 64
        self.layer1, inplanes = _make_resnet_layer(inplanes, 64, 3, stride=(1, 1, 1))
        self.layer2, inplanes = _make_resnet_layer(inplanes, 128, 4, stride=(2, 2, 2))
        self.layer3, inplanes = _make_resnet_layer(inplanes, 256, 6, stride=(2, 2, 2))
        self.layer4, inplanes = _make_resnet_layer(inplanes, 320, 3, stride=(2, 2, 2))
        self._layers: list[torch.Tensor] = []

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, Conv3dWD)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, (nn.InstanceNorm3d,)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._layers = []
        x = self.nonlin(self.norm1(self.conv1(x)))
        self._layers.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        self._layers.append(x)
        x = self.layer2(x)
        self._layers.append(x)
        x = self.layer3(x)
        self._layers.append(x)
        x = self.layer4(x)
        self._layers.append(x)
        return x

    def get_layers(self) -> list[torch.Tensor]:
        return self._layers


class _SharedEncoder(nn.Module):
    """Legacy ``U_Res3D_enc``: ResNet50 backbone + ASPP bottleneck, run once
    per modality via the batch-folding trick (see ``LCKD.forward``)."""

    def __init__(self) -> None:
        super().__init__()
        self.asppreduce = nn.Sequential(conv3x3x3(1280, 256, kernel_size=1), norm_layer(256), activation_layer())
        self.aspp = ASPP(256, 256)
        self.backbone = _ResNet50Backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone(x)
        layers = self.backbone.get_layers()
        reduced = self.asppreduce(layers[-1])
        return self.aspp(reduced)


class _SharedDecoder(nn.Module):
    """Legacy ``U_Res3D_dec``, with the final head changed from a 3-channel
    BraTS-region (ET/WT/TC) sigmoid output to a standard softmax
    ``num_cls``-channel prediction (see module docstring / docs/training.md
    for why)."""

    def __init__(self, num_cls: int) -> None:
        super().__init__()
        self.upsamplex2 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")

        self.shortcut_conv3 = nn.Sequential(conv3x3x3(256 * num_modals * 4, 256, kernel_size=1), norm_layer(256), activation_layer())
        self.shortcut_conv2 = nn.Sequential(conv3x3x3(128 * num_modals * 4, 128, kernel_size=1), norm_layer(128), activation_layer())
        self.shortcut_conv1 = nn.Sequential(conv3x3x3(64 * num_modals * 4, 64, kernel_size=1), norm_layer(64), activation_layer())
        self.shortcut_conv0 = nn.Sequential(conv3x3x3(64 * num_modals, 32, kernel_size=1), norm_layer(32), activation_layer())

        self.transposeconv_stage3 = nn.ConvTranspose3d(256 * num_modals, 256, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, bias=False)

        self.stage3_de = BasicBlock(256, 256)
        self.stage2_de = BasicBlock(128, 128)
        self.stage1_de = BasicBlock(64, 64)
        self.stage0_de = BasicBlock(32, 32)

        self.cls_conv = nn.Conv3d(32, num_cls, kernel_size=1, bias=False)

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, Conv3dWD, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, (nn.InstanceNorm3d,)):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, layers: list[torch.Tensor]) -> torch.Tensor:
        x = self.transposeconv_stage3(x)
        x = x + self.shortcut_conv3(layers[-2])
        x = self.stage3_de(x)

        x = self.transposeconv_stage2(x)
        x = x + self.shortcut_conv2(layers[-3])
        x = self.stage2_de(x)

        x = self.transposeconv_stage1(x)
        x = x + self.shortcut_conv1(layers[-4])
        x = self.stage1_de(x)

        x = self.transposeconv_stage0(x)
        x = x + self.shortcut_conv0(layers[-5])
        x = self.stage0_de(x)

        logits = self.cls_conv(x)
        return self.upsamplex2(logits)


class LCKD(AbstractModel):
    """Port of legacy LCKD's live model, ``DualNet`` (the repo folder is
    named "LCKD"; ``train.py`` never imports anything else).

    Architecture: a *single* weight-shared, per-modality ResNet50+ASPP
    encoder/U-Net decoder (not a teacher/student pair of full networks like
    MST-KDNet). All 4 modalities are batch-folded through the same encoder
    (``[N, 4, D, H, W] -> [N*4, 1, D, H, W]``), producing one 256-channel
    bottleneck feature map per modality. Missing modalities are zero-filled
    before encoding.

    The namesake "KD" mechanism is a **feature-level self-distillation
    regularizer**, not a second network: legacy pulls each present
    modality's bottleneck feature (detached "teacher") toward every other
    present modality's feature (L1), where the *set of teacher modalities*
    is picked dynamically by re-running validation-set Dice scoring every
    `val_pred_every` training iterations and taking the modality with the
    best score per region. That curriculum has no natural place in
    MissingBench's per-batch trainer contract (it needs a mid-training,
    non-differentiable validation loop wired into the optimization step),
    so this port replaces it with a fixed, symmetric scheme: **every
    present modality distills into every other present modality, equally**,
    every training step. This keeps the actual mechanism (cross-modal
    feature-consistency regularization to compensate for missing
    modalities) without depending on an external Dice-driven teacher
    schedule.

    After optional distillation, any modality missing for a given sample
    has its bottleneck feature replaced by the mean of that sample's
    present-modality features (legacy's "fts fill in process"); all 4
    modality feature maps are then channel-concatenated and passed through
    one shared U-Net decoder (with skip connections similarly
    channel-concatenated across modalities from the backbone's
    intermediate layers).

    Bugs/deviations fixed relative to legacy:
    - Legacy applies a single global ``mode`` string (which modalities are
      "available") to an entire training batch; MissingBench's masks are
      per-sample ``[B, 4]``. This port respects the true per-sample mask
      for both the missing-modality fill-in and the distillation loss.
    - Legacy's segmentation head outputs 3 BraTS-region (ET/WT/TC) channels
      trained with sigmoid BCE + Dice, hardcoding an assumption that never
      generalizes to a disjoint ``num_cls``-class one-hot target (breaking
      for BraTS25's 5 classes). This port's head outputs ``num_cls``
      channels with a softmax, reusing ``softmax_weighted_loss``/
      ``dice_loss`` from ``losses/imfuse.py`` like every other model here.
    - Legacy's optional self-/cross-attention bottleneck refinement
      (``nn.MultiheadAttention``) is constructed but never enabled by
      legacy's own ``train.py`` (always called with ``self_att=False,
      cross_att=False``) — dead code, not ported.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.shared_enc = _SharedEncoder()
        self.shared_dec = _SharedDecoder(num_cls)
        self.softmax = nn.Softmax(dim=1)
        self.kd_loss_fn = nn.L1Loss()
        self.is_training = False

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x, mask = self._remap_input_order(x, mask)
        batch, channels, depth, height, width = x.shape

        masked = x * mask.to(x.dtype).view(batch, channels, 1, 1, 1)
        flat = masked.reshape(batch * channels, 1, depth, height, width)

        shared_ft = self.shared_enc(flat)
        backbone_layers = self.shared_enc.backbone.get_layers()
        cat_layers = [
            layer.view(batch, channels * layer.shape[1], layer.shape[2], layer.shape[3], layer.shape[4])
            for layer in backbone_layers
        ]

        feat = shared_ft.view(batch, channels, *shared_ft.shape[1:])
        fused = feat.clone()
        kd_loss = x.new_tensor(0.0)

        for n in range(batch):
            present = [m for m in range(channels) if bool(mask[n, m])]
            if not present:
                continue

            if self.is_training and len(present) > 1:
                for teacher in present:
                    for student in present:
                        kd_loss = kd_loss + self.kd_loss_fn(feat[n, teacher].detach(), feat[n, student])

            missing = [m for m in range(channels) if m not in present]
            if missing:
                fill = feat[n, present].mean(dim=0)
                for m in missing:
                    fused[n, m] = fill

        fused_ft = fused.view(batch, channels * fused.shape[2], fused.shape[3], fused.shape[4], fused.shape[5])
        logits = self.shared_dec(fused_ft, cat_layers)
        probs = self.softmax(logits)

        if not self.is_training:
            return probs
        return probs, kd_loss

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        num_cls = self.shared_dec.cls_conv.out_channels
        prediction = torch.zeros(images.size(0), num_cls, height, width, depth, device=images.device)
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size]
                    patch_pred = self(patch, mask)
                    prediction[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size] += patch_pred
                    weight[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size] += 1
        return prediction / weight

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                f"Model prediction expects spatial dimensions to be at least {input_patch_size}, got {size}"
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
            raise RuntimeError(f"LCKD expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"LCKD expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = LCKD


__all__ = ["LCKD", "Model"]
