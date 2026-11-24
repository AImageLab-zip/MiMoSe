from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

basic_dims = 8
num_modals = 4
input_patch_size = 128

# External pipeline uses [t1c, t1n, t2f, t2w]; SRMNet internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/IMFuse).
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int) -> nn.Module:
    return nn.InstanceNorm3d(planes)


class ConvBlock(nn.Module):
    """Pre-activation conv block: norm -> LeakyReLU -> conv."""

    def __init__(self, in_ch: int, out_ch: int, k_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.norm = normalization(in_ch)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode="reflect", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.activation(self.norm(x)))


def _channel_last(x: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = x.shape
    return x.permute(0, 2, 3, 4, 1).reshape(b, d * h * w, c)


def _channel_first(x: torch.Tensor, d: int, h: int, w: int) -> torch.Tensor:
    b, _, c = x.shape
    return x.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d, h, w = x.shape[-3:]
        flat = _channel_last(x)
        mean = flat.mean(-1, keepdim=True)
        var = flat.var(-1, keepdim=True, unbiased=False)
        normed = (flat - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias
        return _channel_first(normed, d, h, w)


class GatedFeedForward(nn.Module):
    """IFN: a depthwise-conv gated-GELU feedforward, Restormer-style."""

    def __init__(self, dim: int, hidden_dim: int = 340) -> None:
        super().__init__()
        self.project_in = nn.Conv3d(dim, hidden_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv3d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2)
        self.project_out = nn.Conv3d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        head_dim = c // self.num_heads
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.num_heads, head_dim, d * h * w) for t in (q, k, v))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, d, h, w)
        return self.project_out(out)


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        head_dim = c // self.num_heads
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.num_heads, head_dim, d * h * w) for t in (q, k, v))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (k.transpose(-2, -1) @ q) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (v @ attn).reshape(b, c, d, h, w)
        return self.project_out(out)


class GlobalBlock(nn.Module):
    """Restormer-style dual (channel + spatial) self-attention block, applied
    to the bottleneck feature of each modality independently."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm1 = BiasFreeLayerNorm(dim)
        self.channel_attn = ChannelAttention(dim, num_heads)
        self.spatial_attn = SpatialAttention(dim, num_heads)
        self.norm2 = BiasFreeLayerNorm(dim)
        self.ffn = GatedFeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.channel_attn(normed) + self.spatial_attn(normed)
        return x + self.ffn(self.norm2(x))


class AdaptiveFusionBlock(nn.Module):
    """Combines a plain 3x3 conv branch with a second, dynamically-gated 3x3
    conv branch, weighted per-sample by a squeeze-and-excitation-style gate.

    Legacy SRMNet ("dap") used a deformable convolution for the second
    branch; that requires a custom compiled CUDA extension unavailable here,
    so this port uses a second plain conv branch instead, keeping the
    dynamic two-branch gating mechanism intact.
    """

    def __init__(self, in_channel: int, reduction: int = 2) -> None:
        super().__init__()
        self.conv_first = ConvBlock(in_channel * num_modals, in_channel, k_size=1, padding=0)
        self.branch_a = ConvBlock(in_channel, in_channel, k_size=3, padding=1)
        self.branch_b = ConvBlock(in_channel, in_channel, k_size=3, padding=1)
        self.conv_last = ConvBlock(in_channel, in_channel, k_size=1, padding=0)

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(in_channel * num_modals, in_channel // reduction, bias=True)
        self.fc2 = nn.Linear(in_channel // reduction, 2, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels = x.shape[:2]
        squeezed = self.global_avg_pool(x).view(batch_size, channels)
        gate = self.softmax(self.fc2(self.lrelu(self.fc1(squeezed))))

        first = self.conv_first(x)
        branch_a = self.branch_a(first)
        branch_b = self.branch_b(first)
        combined = branch_a * gate[:, 0].view(batch_size, 1, 1, 1, 1) + branch_b * gate[:, 1].view(batch_size, 1, 1, 1, 1)
        return self.conv_last(combined)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = nn.Conv3d(1, basic_dims, kernel_size=3, padding=1, bias=True)
        self.e1_c2 = ConvBlock(basic_dims, basic_dims)
        self.e1_c3 = ConvBlock(basic_dims, basic_dims)

        self.e2_c1 = ConvBlock(basic_dims, basic_dims * 2, stride=2)
        self.e2_c2 = ConvBlock(basic_dims * 2, basic_dims * 2)
        self.e2_c3 = ConvBlock(basic_dims * 2, basic_dims * 2)

        self.e3_c1 = ConvBlock(basic_dims * 2, basic_dims * 4, stride=2)
        self.e3_c2 = ConvBlock(basic_dims * 4, basic_dims * 4)
        self.e3_c3 = ConvBlock(basic_dims * 4, basic_dims * 4)

        self.e4_c1 = ConvBlock(basic_dims * 4, basic_dims * 8, stride=2)
        self.e4_c2 = ConvBlock(basic_dims * 8, basic_dims * 8)
        self.e4_c3 = ConvBlock(basic_dims * 8, basic_dims * 8)

        self.e5_c1 = ConvBlock(basic_dims * 8, basic_dims * 16, stride=2)
        self.e5_c2 = ConvBlock(basic_dims * 16, basic_dims * 16)
        self.e5_c3 = ConvBlock(basic_dims * 16, basic_dims * 16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        return x1, x2, x3, x4, x5


class ReconstructionDecoder(nn.Module):
    """A single shared decoder that reconstructs a modality's own raw image
    from its own encoder features (used only during training)."""

    def __init__(self) -> None:
        super().__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d4_c1 = ConvBlock(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = ConvBlock(basic_dims * 16, basic_dims * 8)
        self.d4_out = ConvBlock(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3_c1 = ConvBlock(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = ConvBlock(basic_dims * 8, basic_dims * 4)
        self.d3_out = ConvBlock(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = ConvBlock(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = ConvBlock(basic_dims * 4, basic_dims * 2)
        self.d2_out = ConvBlock(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = ConvBlock(basic_dims * 2, basic_dims)
        self.d1_c2 = ConvBlock(basic_dims * 2, basic_dims)
        self.d1_out = ConvBlock(basic_dims, basic_dims, k_size=1, padding=0)

        self.image_layer = nn.Conv3d(basic_dims, 1, kernel_size=1, bias=True)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> torch.Tensor:
        de_x5 = self.d4_c1(self.d4(x5))
        de_x4 = self.d4_out(self.d4_c2(torch.cat((de_x5, x4), dim=1)))
        de_x4 = self.d3_c1(self.d3(de_x4))

        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x4, x3), dim=1)))
        de_x3 = self.d2_c1(self.d2(de_x3))

        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x3, x2), dim=1)))
        de_x2 = self.d1_c1(self.d1(de_x2))

        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x2, x1), dim=1)))
        return self.image_layer(de_x1)


class MaskModal(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y.view(batch_size, -1, height, width, depth)


class Decoder(nn.Module):
    def __init__(self, num_cls: int) -> None:
        super().__init__()
        self.d4_c1 = ConvBlock(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = ConvBlock(basic_dims * 16, basic_dims * 8)
        self.d4_out = ConvBlock(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3_c1 = ConvBlock(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = ConvBlock(basic_dims * 8, basic_dims * 4)
        self.d3_out = ConvBlock(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2_c1 = ConvBlock(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = ConvBlock(basic_dims * 4, basic_dims * 2)
        self.d2_out = ConvBlock(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1_c1 = ConvBlock(basic_dims * 2, basic_dims)
        self.d1_c2 = ConvBlock(basic_dims * 2, basic_dims)
        self.d1_out = ConvBlock(basic_dims, basic_dims, k_size=1, padding=0)

        self.seg_d4 = nn.Conv3d(basic_dims * 16, num_cls, kernel_size=1, bias=True)
        self.seg_d3 = nn.Conv3d(basic_dims * 8, num_cls, kernel_size=1, bias=True)
        self.seg_d2 = nn.Conv3d(basic_dims * 4, num_cls, kernel_size=1, bias=True)
        self.seg_d1 = nn.Conv3d(basic_dims * 2, num_cls, kernel_size=1, bias=True)
        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="trilinear", align_corners=True)

        self.fuse5 = AdaptiveFusionBlock(basic_dims * 16)
        self.fuse4 = AdaptiveFusionBlock(basic_dims * 8)
        self.fuse3 = AdaptiveFusionBlock(basic_dims * 4)
        self.fuse2 = AdaptiveFusionBlock(basic_dims * 2)
        self.fuse1 = AdaptiveFusionBlock(basic_dims * 1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        de_x5 = self.fuse5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.fuse4(x4)
        de_x4 = self.d4_out(self.d4_c2(torch.cat((de_x4, de_x5), dim=1)))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.fuse3(x3)
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x3, de_x4), dim=1)))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.fuse2(x2)
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x2, de_x3), dim=1)))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.fuse1(x1)
        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x1, de_x2), dim=1)))

        pred = self.softmax(self.seg_layer(de_x1))
        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SRMNet(AbstractModel):
    """Implementation of SRMNet introduced in [1].

    [1] "SRMNet: Style Recalibration and Modality-aware Network for
        Incomplete Multimodal Brain Tumor Segmentation".

    Each modality is encoded independently; the bottleneck feature of each
    modality is refined by a Restormer-style dual channel/spatial
    self-attention block before missing modalities are zeroed and
    concatenated for the decoder, matching DCSeg/RFNet's masking convention.
    Each decoder stage fuses the concatenated modality features via a
    dynamically-gated two-branch conv block ("adaptive fusion block";
    legacy used a deformable convolution for the second branch — see
    ``AdaptiveFusionBlock``), and produces a multi-scale (deep-supervision)
    prediction. During training only, a single shared reconstruction decoder
    also reconstructs each modality's own raw image from its own encoder
    features, for an auxiliary L1 loss.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls

        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.flair_glo = GlobalBlock(dim=basic_dims * 16)
        self.t1ce_glo = GlobalBlock(dim=basic_dims * 16)
        self.t1_glo = GlobalBlock(dim=basic_dims * 16)
        self.t2_glo = GlobalBlock(dim=basic_dims * 16)

        self.rec = ReconstructionDecoder()
        self.masker = MaskModal()
        self.decoder = Decoder(num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor], torch.Tensor]:
        x, mask = self._remap_input_order(x, mask)

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4])

        flair_g5 = self.flair_glo(flair_x5)
        t1ce_g5 = self.t1ce_glo(t1ce_x5)
        t1_g5 = self.t1_glo(t1_x5)
        t2_g5 = self.t2_glo(t2_x5)

        m1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        m2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        m3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        m4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        m5 = self.masker(torch.stack((flair_g5, t1ce_g5, t1_g5, t2_g5), dim=1), mask)

        pred, preds = self.decoder(m1, m2, m3, m4, m5)

        if not self.is_training:
            return pred

        recs = [
            self.rec(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5),
            self.rec(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5),
            self.rec(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5),
            self.rec(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5),
        ]
        # x is in remapped [flair, t1ce, t1, t2] order, matching recs.
        return pred, preds, recs, x

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        was_training = self.is_training
        self.is_training = False
        try:
            _, _, height, width, depth = images.shape
            if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
                return self(images, mask)

            h_starts = self._window_starts(height)
            w_starts = self._window_starts(width)
            d_starts = self._window_starts(depth)
            prediction = torch.zeros(images.size(0), self.num_cls, height, width, depth, device=images.device)
            weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

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
        finally:
            self.is_training = was_training

    @staticmethod
    def _window_starts(size: int) -> list[int]:
        if size < input_patch_size:
            raise RuntimeError(
                f"Model prediction expects spatial dimensions to be at least {input_patch_size}, got {size}"
            )
        if size == input_patch_size:
            return [0]

        stride = int(input_patch_size * 0.5)
        starts = list(range(0, max(size - input_patch_size, 0), stride))
        last_start = size - input_patch_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _remap_input_order(
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"SRMNet expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"SRMNet expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = SRMNet


__all__ = ["SRMNet", "Model"]
