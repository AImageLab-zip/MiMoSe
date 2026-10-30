from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121
from torchvision.models.densenet import _DenseBlock

from mimose.models.abstract_model import AbstractModel

num_modals = 4
# The 2D network downsamples by 32 in its DenseNet121 prior encoder.
spatial_multiple = 32

# External pipeline uses [t1c, t1n, t2f, t2w]; legacy NAP-SCAF stacks
# [flair, t1, t1ce, t2].
DATASET_MODALITY_ORDER = (2, 1, 0, 3)


class _UpTransition(nn.Sequential):
    """Legacy decoder ``_Transition``: DenseNet's transition with the pooling swapped for 2x upsampling."""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.Upsample(scale_factor=2.0))


class _PriorEncoder(nn.Module):
    """DenseNet121 feature extractor (4-channel stem) followed by a mu/logvar projection."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.features = densenet121(weights=weights).features
        self.features.conv0 = nn.Conv2d(num_modals, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Conv2d(1024, 2 * 1024, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.fc(self.features(x)).chunk(2, dim=1)
        return mu, logvar


class _PriorDecoder(nn.Module):
    def __init__(self, cdim: int = num_modals, zdim: int = 1024) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(zdim, 1024, 3, 1, 1), nn.ReLU(True))
        self.decoder = nn.Sequential(
            _UpTransition(1024, 512),
            _DenseBlock(num_layers=16, num_input_features=512, bn_size=4, growth_rate=32, drop_rate=0),
            _UpTransition(1024, 256),
            _DenseBlock(num_layers=8, num_input_features=256, bn_size=4, growth_rate=32, drop_rate=0),
            _UpTransition(512, 128),
            _DenseBlock(num_layers=4, num_input_features=128, bn_size=4, growth_rate=32, drop_rate=0),
            _UpTransition(256, 64),
            _DenseBlock(num_layers=2, num_input_features=64, bn_size=4, growth_rate=32, drop_rate=0),
            _UpTransition(128, 32),
            nn.Conv2d(32, cdim, 3, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.fc(z))


class Conv3x3(nn.Module):
    def __init__(self, inc: int, outc: int, stride: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inc, outc, 3, stride, 1)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class NAPPath(nn.Module):
    """Non-local Adaptive Prior: variational reconstruction + a small U-Net giving multi-scale prior features.

    As in legacy, the VAE reconstruction runs under ``torch.no_grad()`` and no
    loss ever reaches it, so the prior encoder/decoder keep their initial
    weights (ImageNet DenseNet121 features with a fresh 4-channel stem, and a
    randomly initialised decoder).
    """

    def __init__(self, midc: int, stages: int, pretrained: bool) -> None:
        super().__init__()
        self.vae = nn.ModuleDict({"enc": _PriorEncoder(pretrained), "dec": _PriorDecoder()})
        self.inconv = Conv3x3(num_modals, midc, 1)
        self.enc = nn.ModuleList()
        stagec = midc
        for _ in range(stages):
            self.enc.append(Conv3x3(stagec, stagec * 2, 2))
            stagec *= 2
        self.dec = nn.ModuleList()
        for _ in range(stages):
            self.dec.append(nn.ModuleList([nn.ConvTranspose2d(stagec, stagec // 2, 2, 2), Conv3x3(stagec, stagec // 2, 1)]))
            stagec //= 2

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.vae["enc"](x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.vae["dec"](z)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.inconv(self.reconstruct(x))
        enc_feas = []
        for layer in self.enc:
            x = layer(x)
            enc_feas.append(x)
        fea = enc_feas[-1]
        dec_feas = []
        for i, (up, merge) in enumerate(self.dec):
            fea = up(fea)
            if i < len(enc_feas) - 1:
                fea = merge(torch.cat([fea, enc_feas[-2 - i]], 1))
            dec_feas.append(fea)
        return dec_feas


class Attention(nn.Module):
    """Legacy channel-wise (transposed) self-attention with a depthwise positional embedding."""

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        # [b, n, h*d] -> [b, h, d, n]
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.dim_head).permute(0, 2, 3, 1)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q = F.normalize(self._heads(self.to_q(x)), dim=-1, p=2)
        k = F.normalize(self._heads(self.to_k(x)), dim=-1, p=2)
        v = self._heads(self.to_v(x))
        attn = (k @ q.transpose(-2, -1) * self.rescale).softmax(dim=-1)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        return out_c + self.pos_emb(x_in)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads)
        self.ff = FeedForward(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x) + x
        return self.ff(x) + x


class CERA(nn.Module):
    """Contrast-Enhanced Region Attention: multi-rate depthwise convs plus a local-contrast cue."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.local1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.local2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels, bias=False)
        self.local3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, groups=in_channels, bias=False)
        self.bg = nn.AvgPool2d(7, stride=1, padding=3)
        self.fuse = nn.Sequential(nn.Conv2d(in_channels * 4, in_channels, 1, bias=False), nn.BatchNorm2d(in_channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contrast = (x - self.bg(x)).abs()
        att = self.fuse(torch.cat([self.local1(x), self.local2(x), self.local3(x), contrast], dim=1))
        return x * att + x


class SWG(nn.Module):
    """Similarity-Weighted Gating: attenuates skip features by their cosine agreement with the prior."""

    def __init__(self, gch: int, xch: int, midc: int, alpha_init: float = 0.1) -> None:
        super().__init__()
        self.q = nn.Conv2d(gch, midc, 1, bias=False)
        self.k = nn.Conv2d(xch, midc, 1, bias=False)
        init = torch.log(torch.tensor(alpha_init) / (1 - torch.tensor(alpha_init)))
        self.logit_alpha = nn.Parameter(init.view(1, 1, 1, 1))

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.q(g), dim=1, eps=1e-6)
        k = F.normalize(self.k(x), dim=1, eps=1e-6)
        sim = ((q * k).sum(dim=1, keepdim=True) + 1.0) * 0.5
        return x * (1.0 - torch.sigmoid(self.logit_alpha) * sim)


class SWCGate(nn.Module):
    def __init__(self, gch: int, xch: int, midc: int) -> None:
        super().__init__()
        self.gconv = nn.Conv2d(gch, midc, 1, bias=False)
        self.xconv = nn.Conv2d(xch, midc, 1, bias=False)
        self.psi = nn.Conv2d(midc, 1, 1, bias=False)
        self.cera = CERA(xch)
        self.swg = SWG(gch, xch, midc, alpha_init=0.1)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        hint = torch.sigmoid(self.psi(torch.relu(-self.gconv(g) * self.xconv(x))))
        return self.swg(g, self.cera(x)) * hint


class _NAPSCAF2D(nn.Module):
    """Legacy ``NAP_SCAF`` 2D network: per-modality encoders, transformer fusion, prior-gated decoder."""

    def __init__(self, outc: int, midc: int, stages: int, gate_passes: int, pretrained_prior: bool) -> None:
        super().__init__()
        if gate_passes < 2:
            raise ValueError("NAP-SCAF needs gate_passes >= 2 (pre- and post-merge calibration)")
        self.stages = stages
        self.gate_passes = gate_passes
        self.nap_path = NAPPath(midc, stages, pretrained_prior)
        self.inconvs = nn.ModuleList([Conv3x3(1, midc, 1) for _ in range(num_modals)])
        self.encs = nn.ModuleList()
        for _ in range(num_modals):
            enc = nn.ModuleList()
            stagec = midc
            for _ in range(stages):
                enc.append(Conv3x3(stagec, stagec * 2, 2))
                stagec *= 2
            self.encs.append(enc)
        self.fusion = nn.ModuleList()
        stagec = midc
        for _ in range(stages):
            stagec *= 2
            self.fusion.append(
                nn.Sequential(
                    Transformer(dim=num_modals * stagec, dim_head=midc, heads=stagec // midc),
                    Conv3x3(num_modals * stagec, stagec, 1),
                )
            )
        self.dec = nn.ModuleList()
        self.gates = nn.ModuleList()
        for _ in range(stages):
            self.dec.append(nn.ModuleList([nn.ConvTranspose2d(stagec, stagec // 2, 2, 2), Conv3x3(stagec, stagec // 2, 1)]))
            stagec //= 2
            self.gates.append(SWCGate(gch=stagec, xch=stagec, midc=stagec // 2))
        # Legacy builds an output head per decoder stage but only returns the last one.
        self.convout = nn.Conv2d(stagec, outc, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nap_feats = self.nap_path(x)
        enc_stacks = []
        for i, xi in enumerate(torch.chunk(x, num_modals, 1)):
            xi = self.inconvs[i](xi)
            feas = []
            for layer in self.encs[i]:
                xi = layer(xi)
                feas.append(xi)
            enc_stacks.append(feas)
        fused = [self.fusion[k](torch.cat([feas[k] for feas in enc_stacks], 1)) for k in range(self.stages)]

        fea = fused[-1]
        for i, (up, merge) in enumerate(self.dec):
            fea = self.gates[i](nap_feats[i], up(fea))
            if i < len(fused) - 1:
                fea = merge(torch.cat([fea, fused[-2 - i]], 1))
            for _ in range(self.gate_passes - 1):
                fea = self.gates[i](nap_feats[i], fea)
        return self.convout(fea)


class NAPSCAF(AbstractModel):
    """Implementation of NAP-SCAF introduced in [1], applied slice-wise to 3D volumes.

    Legacy NAP-SCAF is a 2D network trained on 256x256 axial PNG slices. This
    port keeps the 2D network unchanged and runs it on the axial slices of
    MiMoSe's 3D inputs: in training mode (``is_training``) ``forward`` takes a
    batch of [N, 4, H, W] slices drawn from the 3D patches and returns logits,
    while at inference ``forward``/``predict`` split a [B, 4, H, W, D] volume
    into its D axial slices, zero-pad H and W to a multiple of 32 (or, with
    ``inference_size``, resize them to legacy's 256x256 and the logits back),
    and stack the per-slice softmax maps back into a volume.
    Missing modalities are zero-filled, as in legacy's evaluation.

    Port deviations (see docs/training.md): the three discarded intermediate
    output heads are not built, ``einops`` is replaced by plain reshapes, and
    the DenseNet121 classifier (unused in legacy's forward) is not kept.

    [1] NAP-SCAF: https://github.com/mh-zhou/NAP-SCAF
    """

    def __init__(
        self,
        num_cls: int = 4,
        mid_channels: int = 16,
        stages: int = 4,
        gate_passes: int = 2,
        pretrained_prior: bool = True,
        slice_chunk: int = 32,
        inference_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.slice_chunk = slice_chunk
        # When set, inference resizes each slice to inference_size^2 (legacy feeds 240x240 BraTS
        # slices resized to 256x256) and resizes the logits back; otherwise slices are zero-padded.
        if inference_size is not None and int(inference_size) % spatial_multiple:
            raise ValueError(f"inference_size must be a multiple of {spatial_multiple}, got {inference_size}")
        self.inference_size = None if inference_size is None else int(inference_size)
        self.net = _NAPSCAF2D(
            outc=num_cls, midc=mid_channels, stages=stages, gate_passes=gate_passes, pretrained_prior=pretrained_prior
        )
        self.is_training = False

    def forward_slices(self, slices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Logits for a batch of 2D slices [N, 4, H, W] with per-slice modality masks [N, 4]."""
        slices, mask = self._remap_input_order(slices, mask)
        masked = slices * mask.view(mask.size(0), num_modals, 1, 1).to(slices.dtype)
        return self.net(masked)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Training (``is_training``): logits for 2D slices [N, 4, H, W]. Otherwise: probabilities for
        [B, 4, H, W, D] volumes, computed slice by slice."""
        if self.is_training:
            if x.ndim != 4:
                raise RuntimeError(f"NAPSCAF trains on [N, {num_modals}, H, W] slices, got {tuple(x.shape)}")
            return self.forward_slices(x, mask)
        if x.ndim != 5:
            raise RuntimeError(f"NAPSCAF expects a [B, {num_modals}, H, W, D] volume, got {tuple(x.shape)}")
        batch, channels, height, width, depth = x.shape

        # [B, C, H, W, D] -> [B*D, C, H, W]
        slices = x.permute(0, 4, 1, 2, 3).reshape(batch * depth, channels, height, width)
        slice_mask = mask.repeat_interleave(depth, dim=0)
        if self.inference_size is None:
            slices = F.pad(slices, (0, (-width) % spatial_multiple, 0, (-height) % spatial_multiple))

        def slice_logits(chunk: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
            if self.inference_size is None:
                return self.forward_slices(chunk, chunk_mask)[:, :, :height, :width]
            size = (self.inference_size, self.inference_size)
            chunk = F.interpolate(chunk, size=size, mode="bilinear", align_corners=False)
            logits = self.forward_slices(chunk, chunk_mask)
            return F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)

        probabilities = torch.cat(
            [
                slice_logits(slices[start : start + self.slice_chunk], slice_mask[start : start + self.slice_chunk])
                .softmax(dim=1)
                for start in range(0, slices.size(0), self.slice_chunk)
            ]
        )
        return probabilities.reshape(batch, depth, self.num_cls, height, width).permute(0, 2, 3, 4, 1)

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Slice-wise inference covers the whole in-plane extent; no sliding window is needed.
        return self(images, mask)

    @staticmethod
    def _remap_input_order(images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"NAPSCAF expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"NAPSCAF expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = NAPSCAF


__all__ = ["NAPSCAF", "Model"]
