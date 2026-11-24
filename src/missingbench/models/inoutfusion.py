from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from missingbench.models.abstract_model import AbstractModel

feature_maps = 8
levels = 4
patch_dim = 8
num_modals = 4
input_patch_size = 128

NEG_INF = -1.0e9

# External pipeline uses [t1c, t1n, t2f, t2w]; legacy InOutFusion's default
# `--selected_modal` is [t1c, t1n, t2w, t2f]. Internal encoder/mask index k
# corresponds directly to bit k of legacy's missing-modality integer code
# (t1c=bit0, t1n=bit1, t2w=bit2, t2f=bit3), so we only need to swap the last
# two external slots to land on legacy's internal order.
DATASET_MODALITY_ORDER = (0, 1, 3, 2)


# ****************************************************************************
# ------------------------------ In-Fusion: per-modality Restormer encoder ---
# ****************************************************************************
class LayerNormChannel(nn.Module):
    """Legacy `LayerNorm(..., 'WithBias')`: LayerNorm over the channel axis of a
    5D volume, applied by moving channels to the last axis and back."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w, d = x.shape[-3:]
        y = rearrange(x, "b c h w d -> b (h w d) c")
        mu = y.mean(-1, keepdim=True)
        sigma = y.var(-1, keepdim=True, unbiased=False)
        y = (y - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
        return rearrange(y, "b (h w d) c -> b c h w d", h=h, w=w, d=d)


class ChannelAttention(nn.Module):
    """Legacy `AttentionBase`: Restormer-style multi-dconv-head *transposed*
    (channel-wise, not spatial) self-attention -- the attention matrix is
    `[heads, C, C]`, independent of the (potentially huge) spatial volume."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv3d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w, d = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (head c) h w d -> b head c (h w d)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w d -> b head c (h w d)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w d -> b head c (h w d)", head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, "b head c (h w d) -> b (head c) h w d", head=self.num_heads, h=h, w=w)
        return self.proj(out)


class GatedMlp(nn.Module):
    """Legacy `Mlp`: gated depthwise-grouped conv feed-forward."""

    def __init__(self, dim: int, expansion: float = 1.0, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(
            hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden, bias=bias, padding_mode="reflect"
        )
        self.project_out = nn.Conv3d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class GlobalFeatureExtraction(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.norm1 = LayerNormChannel(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads)
        self.norm2 = LayerNormChannel(dim)
        self.mlp = GatedMlp(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + x


class LocalFeatureExtraction(nn.Module):
    def __init__(self, dim: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ResBlock3D(dim) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RestormerBlock(nn.Module):
    """Legacy `Restormer_CNN_block`: the "In-Fusion" building block -- fuses a
    global (channel-attention) branch with a local (residual conv) branch via
    a small FFN, giving each per-modality encoder stage both a long-range and
    a local receptive field cheaply (the attention matrix is channel-sized,
    not spatial-sized)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.embed = nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1, bias=False, padding_mode="reflect")
        self.global_feature = GlobalFeatureExtraction(out_dim, num_heads=min(8, out_dim))
        self.local_feature = LocalFeatureExtraction(out_dim)
        self.ffn = nn.Conv3d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False, padding_mode="reflect")
        self.norm = nn.InstanceNorm3d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x1 = self.global_feature(x)
        x2 = self.local_feature(x)
        out = self.ffn(torch.cat((x1, x2), dim=1))
        return self.norm(out)


class RSEncoder(nn.Module):
    """Legacy `RSEncoder`: a 4-level per-modality encoder built from
    `RestormerBlock`s with max-pool downsampling between levels."""

    def __init__(self, in_channels: int = 1, base_channels: int = feature_maps) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stage1 = RestormerBlock(in_channels, channels[0])
        self.stage2 = RestormerBlock(channels[0], channels[1])
        self.stage3 = RestormerBlock(channels[1], channels[2])
        self.stage4 = RestormerBlock(channels[2], channels[3])
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v1 = self.stage1(x)
        v2 = self.stage2(self.down1(v1))
        v3 = self.stage3(self.down2(v2))
        v4 = self.stage4(self.down3(v3))
        return v1, v2, v3, v4


# ****************************************************************************
# ---------------------- Out-Fusion: cross-modality attention fusion ---------
# ****************************************************************************
class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.pow(10000.0, torch.arange(0, dim, 2).float() / dim)
        table = torch.zeros(max_len, dim)
        table[:, 0::2] = torch.sin(position / div_term)
        table[:, 1::2] = torch.cos(position / div_term[: table[:, 1::2].shape[1]])
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table[:, : x.size(1)].to(dtype=x.dtype)


class DUpsampling3D(nn.Module):
    """Legacy `DUpsampling3D`: a channel-to-space ("pixel shuffle") upsampler
    used to project attention maps computed on a coarse (avg-pooled) token
    grid back to a per-modality feature map's native spatial resolution."""

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels * (scale**3), kernel_size=1, bias=False)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        batch, channels, depth, height, width = x.size()
        scale = self.scale

        x = x.permute(0, 4, 3, 2, 1).contiguous().view(batch, width, height, depth * scale, channels // scale)
        x = x.permute(0, 3, 1, 2, 4).contiguous().view(batch, depth * scale, width, height * scale, channels // (scale**2))
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch, depth * scale, height * scale, width * scale, channels // (scale**3))
        return x.permute(0, 4, 1, 2, 3).contiguous()


def _key_mask_bias(mask: torch.Tensor, tokens_per_modality: int) -> torch.Tensor:
    """Build an additive attention bias of shape [B, 1, 1, L] that forbids
    attending to any token belonging to a modality missing for that sample."""
    allowed = mask.repeat_interleave(tokens_per_modality, dim=1)  # [B, L] bool
    bias = torch.zeros_like(allowed, dtype=torch.float32)
    bias = bias.masked_fill(~allowed, NEG_INF)
    return bias[:, None, None, :]


class MaskedSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, channels // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_bias is not None:
            attn = attn + key_bias
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionLayer(nn.Module):
    """One pre-norm masked-self-attention + FFN block over the token sequence."""

    def __init__(self, dim: int, heads: int = 4, mlp_dim: int = 512) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskedSelfAttention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class SpectralGateUnit(nn.Module):
    """Simplified, self-contained stand-in for legacy's vendored
    `AFNO1D_channelfirst` (an Adaptive Fourier Neural Operator block from the
    CVNets/AFFNet package, which required an unrelated `opts`/argparse
    configuration object and is not available in this environment). This
    keeps the same idea -- global token mixing performed in the frequency
    domain via a real FFT along the token axis, followed by a block-diagonal
    2-layer complex MLP and an inverse FFT -- without the vendored plumbing.
    """

    def __init__(self, dim: int, num_blocks: int = 4, hidden_factor: int = 1) -> None:
        super().__init__()
        if dim % num_blocks != 0:
            num_blocks = 1
        self.num_blocks = num_blocks
        self.block_size = dim // num_blocks
        scale = 0.02
        hidden = self.block_size * hidden_factor
        self.w1 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size, hidden))
        self.b1 = nn.Parameter(scale * torch.randn(2, num_blocks, hidden))
        self.w2 = nn.Parameter(scale * torch.randn(2, num_blocks, hidden, self.block_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))
        self.sparsity_threshold = 0.01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = x
        batch, tokens, channels = x.shape
        dtype = x.dtype
        x = x.float()

        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(batch, x.shape[1], self.num_blocks, self.block_size)

        o1_real = F.relu(
            torch.einsum("bfki,kio->bfko", x.real, self.w1[0])
            - torch.einsum("bfki,kio->bfko", x.imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag = F.relu(
            torch.einsum("bfki,kio->bfko", x.imag, self.w1[0])
            + torch.einsum("bfki,kio->bfko", x.real, self.w1[1])
            + self.b1[1]
        )
        o2_real = (
            torch.einsum("bfki,kio->bfko", o1_real, self.w2[0])
            - torch.einsum("bfki,kio->bfko", o1_imag, self.w2[1])
            + self.b2[0]
        )
        o2_imag = (
            torch.einsum("bfki,kio->bfko", o1_imag, self.w2[0])
            + torch.einsum("bfki,kio->bfko", o1_real, self.w2[1])
            + self.b2[1]
        )
        out = torch.stack([o2_real, o2_imag], dim=-1)
        out = F.softshrink(out, lambd=self.sparsity_threshold)
        out = torch.view_as_complex(out)
        out = out.reshape(batch, out.shape[1], channels)
        out = torch.fft.irfft(out, n=tokens, dim=1, norm="ortho")
        return out.to(dtype) + bias


class SpectralLayer(nn.Module):
    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mix = SpectralGateUnit(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DeepFusionMixer(nn.Module):
    """Legacy `AF_3d`: combines a spectral-mixing branch and a full
    self-attention branch via a learned per-token sigmoid gate. Used at the
    network bottleneck (`method='TF', trans='ff'` in legacy)."""

    def __init__(self, dim: int, heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        mlp_dim = dim * 4
        self.spectral_branch = nn.ModuleList([SpectralLayer(dim, mlp_dim) for _ in range(num_layers)])
        self.attn_branch = nn.ModuleList([AttentionLayer(dim, heads=heads, mlp_dim=mlp_dim) for _ in range(num_layers)])
        self.gate = nn.Linear(dim * 2, 1)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor | None = None) -> torch.Tensor:
        x1 = x
        for layer in self.spectral_branch:
            x1 = layer(x1)
        x2 = x
        for layer in self.attn_branch:
            x2 = layer(x2, key_bias)
        gate = torch.sigmoid(self.gate(torch.cat((x1, x2), dim=-1)))
        return gate * x1 + (1.0 - gate) * x2


class ShallowFusionMixer(nn.Module):
    """Legacy `LTEncoderLayer` (`method='la'` in legacy): a single masked
    self-attention + FFN block used at the cheaper skip-connection levels.
    Legacy wraps a pip `local-attention` package (causal, window_size=128)
    here, but causal windowing has no natural meaning for volumetric tokens
    with no sequential order -- an artifact of reusing an NLP-oriented
    library without adapting its settings. This port replaces it with plain
    (non-causal, full-sequence) masked self-attention, which is also cheap
    at this token-sequence length (<= `4 * patch_dim**3` tokens)."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.layer = AttentionLayer(dim, heads=heads, mlp_dim=dim * 4)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor | None = None) -> torch.Tensor:
        return self.layer(x, key_bias)


class OutFusion(nn.Module):
    """Legacy `TF_3D`: tokenizes each present modality's feature map (via
    average pooling to a fixed `patch_dim**3` grid), mixes the concatenated
    per-modality token sequence with `mixer`, reprojects the mixed tokens
    back to per-modality attention maps (via `DUpsampling3D`), and combines
    the *original* (full-resolution) per-modality feature maps with a
    softmax-over-modality attention weighting.

    Legacy runs this only over the present modalities (a variable-length
    list, since legacy's dataloader always uses batch size 1). MissingBench uses
    real per-sample `[B, 4]` masks with batch size > 1, where different
    samples can have different present/missing patterns, so a variable
    number of tokens per sample cannot be batched. This port always
    tokenizes all 4 modality slots (zero-filling the token blocks of
    modalities missing for that sample, consistent with the rest of
    MissingBench's `MaskModal`-style convention) and additionally masks attention
    keys and the final modality-softmax logits so missing modalities are
    excluded from both the token-mixing step and the fused output --
    reproducing legacy's per-sample exclusion behaviour under batching.
    """

    def __init__(self, channels: int, spatial_size: int, variant: str, heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        self.scale = spatial_size // patch_dim
        self.avgpool = nn.AdaptiveAvgPool3d(patch_dim)
        self.upsample = DUpsampling3D(channels, self.scale) if self.scale > 1 else None
        self.pos_encoding = PositionalEncoding(channels, max_len=(patch_dim**3) * num_modals)
        self.dropout = nn.Dropout(0.1)
        if variant == "deep":
            self.mixer: nn.Module = DeepFusionMixer(channels, heads=heads)
        elif variant == "shallow":
            self.mixer = ShallowFusionMixer(channels, heads=heads)
        else:
            raise ValueError(f"unknown OutFusion variant: {variant}")

    def forward(self, features: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        batch = features[0].size(0)
        tokens_per_modality = patch_dim**3

        token_blocks = []
        for modal_index in range(num_modals):
            pooled = self.avgpool(features[modal_index])
            pooled = rearrange(pooled, "b c h w d -> b (h w d) c")
            present = mask[:, modal_index].to(pooled.dtype).view(batch, 1, 1)
            token_blocks.append(pooled * present)
        tokens = torch.cat(token_blocks, dim=1)
        tokens = self.dropout(self.pos_encoding(tokens))

        key_bias = _key_mask_bias(mask, tokens_per_modality)
        mixed = self.mixer(tokens, key_bias)

        attn_logits = []
        for modal_index in range(num_modals):
            chunk = mixed[:, modal_index * tokens_per_modality : (modal_index + 1) * tokens_per_modality, :]
            chunk = rearrange(chunk, "b (h w d) c -> b c h w d", h=patch_dim, w=patch_dim, d=patch_dim)
            if self.upsample is not None:
                chunk = self.upsample(chunk)
            attn_logits.append(chunk)
        logits = torch.stack(attn_logits, dim=0)  # [num_modals, B, C, H, W, D]

        modality_bias = (~mask).permute(1, 0).to(logits.dtype) * NEG_INF
        logits = logits + modality_bias[:, :, None, None, None, None]
        weights = logits.softmax(dim=0)

        fused = features[0].new_zeros(features[0].shape)
        for modal_index in range(num_modals):
            fused = fused + features[modal_index] * weights[modal_index]
        return fused


# ****************************************************************************
# ------------------------------------- Decoder -------------------------------
# ****************************************************************************
class DecoderConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class Decoder(nn.Module):
    """Legacy `UNetDecoder`: 3 upsample + concat-skip + conv-block stages
    followed by a 1x1 segmentation head."""

    def __init__(self, num_cls: int) -> None:
        super().__init__()
        channels = [feature_maps * (2**i) for i in range(levels)]  # [8, 16, 32, 64]
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.block3 = DecoderConvBlock(channels[3] + channels[2], channels[2])
        self.block2 = DecoderConvBlock(channels[2] + channels[1], channels[1])
        self.block1 = DecoderConvBlock(channels[1] + channels[0], channels[0])
        self.seg_layer = nn.Conv3d(channels[0], num_cls, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        skip1, skip2, skip3 = skips
        x = self.block3(torch.cat((skip3, self.upsample(bottleneck)), dim=1))
        x = self.block2(torch.cat((skip2, self.upsample(x)), dim=1))
        x = self.block1(torch.cat((skip1, self.upsample(x)), dim=1))
        return self.softmax(self.seg_layer(x))


# ****************************************************************************
# ------------------------------------- Model ---------------------------------
# ****************************************************************************
class InOutFusion(AbstractModel):
    """Port of legacy InOutFusion's `RsInOut_U_Hemis3D` -- the only model
    class actually wired up by `train.py` (`--model_name` defaults to
    `RsInOut_U_Hemis3D`, and the loss dispatch in `Solver.train`/`Solver.val`
    is keyed on `'Hemis3D' in self.model_name`, which selects the plain
    `U_Hemis_loss`, a pure weighted multi-class Dice loss with no
    cross-entropy term and no deep supervision). Legacy's `Network_InOut.py`
    also defines `U_Hemis3D`/`TF_U_Hemis3D`/`AF_U_Hemis3D`/`InOut_U_Hemis3D`,
    and `Network_RMBTS.py`/`Network_LMCR.py` define entirely separate RMBTS/
    LMCR model families with their own losses -- none of these are ever
    selected by the default training command and are not ported.

    Architecture ("Hierarchical In-Out Fusion"):
    - **In-Fusion**: 4 independent per-modality encoders (`RSEncoder`), each
      a 4-level stack of `RestormerBlock`s (a Restormer-style *channel*
      self-attention branch, cheap regardless of spatial resolution, fused
      with a local residual-conv branch via a small conv FFN).
    - **Out-Fusion**: at the bottleneck (deepest, coarsest level) and at
      each of the 3 encoder skip levels, `OutFusion` tokenizes every
      modality's feature map (avg-pooled to a fixed `patch_dim**3` grid),
      mixes the per-modality tokens with a cross-modality mixer, and
      reprojects the mixed tokens into a per-modality spatial attention map
      (`DUpsampling3D`) that is used to softmax-combine the original
      (full-resolution) per-modality features. The bottleneck uses a
      "deep" mixer combining two branches (self-attention and a frequency-
      domain spectral-gating unit) via a learned gate; the 3 skip levels
      use a cheaper "shallow" mixer (a single self-attention layer),
      matching legacy's `method='TF'` vs. `method='la'` distinction.
    - A single U-Net-style decoder consumes the fused bottleneck + 3 fused
      skip connections and outputs per-voxel class probabilities. There is
      no auxiliary per-modality decoder and no deep supervision.

    Bugs found and fixed relative to legacy:
    - Legacy resolves the batch's missing-modality pattern from `m_d[0]`
      only (a scalar, since legacy's dataloader hard-asserts
      `batch_size == 1`) and additionally only ever encodes the *present*
      modalities (a variable-length list), which cannot be batched with
      MissingBench's true per-sample `[B, 4]` masks. This port always runs all 4
      encoders and instead zero-fills token blocks plus masks attention
      keys and the final modality-softmax logits for modalities missing on
      a *per-sample* basis (see `OutFusion`), which is equivalent to
      legacy's exclusion behaviour under `batch_size == 1` and generalizes
      correctly to batched, per-sample masks.
    - Legacy's `general_dice_loss` hardcodes per-class Dice weights
      `[0.1, 0.2, 0.3, 0.4]`, tied to exactly 4 classes; this silently
      breaks for BraTS25's 5 classes (the same `num_cls`-vs.-class-count
      class of bug found in SRMNet/MMMViT/IMS2Trans/RFL, though
      here it lives in the loss rather than a module constructor). See
      `InOutFusionLoss` for the generalized per-class weighting.

    Dependency simplifications (legacy code paths that are not portable to
    this environment, replaced with self-contained plain-torch equivalents;
    see `SpectralGateUnit` and `ShallowFusionMixer` docstrings for details):
    - Legacy's bottleneck mixer (`AF_3d`) uses a vendored CVNets/AFFNet
      `AFNO1D_channelfirst` block requiring an unrelated `opts` argparse
      configuration object; replaced with a self-contained real-FFT
      spectral-gating unit preserving the same "frequency-domain global
      mixing" idea.
    - Legacy's skip-level mixer (`LTEncoderLayer`) wraps a pip
      `local-attention` package configured as *causal* with
      `window_size=128`; replaced with plain (non-causal, full-sequence)
      masked self-attention, since causal windowing has no meaning for
      volumetric tokens with no sequential order.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.encoders = nn.ModuleList([RSEncoder(in_channels=1, base_channels=feature_maps) for _ in range(num_modals)])

        channels = [feature_maps * (2**i) for i in range(levels)]  # [8, 16, 32, 64]
        spatial = [input_patch_size // (2**i) for i in range(levels)]  # [128, 64, 32, 16]

        self.fusion1 = OutFusion(channels[0], spatial[0], variant="shallow")
        self.fusion2 = OutFusion(channels[1], spatial[1], variant="shallow")
        self.fusion3 = OutFusion(channels[2], spatial[2], variant="shallow")
        self.fusion4 = OutFusion(channels[3], spatial[3], variant="deep")

        self.decoder = Decoder(num_cls=num_cls)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x, mask = self._remap_input_order(x, mask)

        v1s, v2s, v3s, v4s = [], [], [], []
        for modal_index in range(num_modals):
            v1, v2, v3, v4 = self.encoders[modal_index](x[:, modal_index : modal_index + 1])
            v1s.append(v1)
            v2s.append(v2)
            v3s.append(v3)
            v4s.append(v4)

        fused_v1 = self.fusion1(v1s, mask)
        fused_v2 = self.fusion2(v2s, mask)
        fused_v3 = self.fusion3(v3s, mask)
        fused_bottleneck = self.fusion4(v4s, mask)

        seg = self.decoder(fused_bottleneck, [fused_v1, fused_v2, fused_v3])

        if not self.is_training:
            return seg
        return seg, (), ()

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.decoder.seg_layer.out_channels,
            height,
            width,
            depth,
            device=images.device,
        )
        weight = torch.zeros(images.size(0), 1, height, width, depth, device=images.device)

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size]
                    patch_pred = self(patch, mask)
                    prediction[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ] += patch_pred
                    weight[
                        :, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size
                    ] += 1
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
            raise RuntimeError(f"InOutFusion expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"InOutFusion expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = InOutFusion


__all__ = ["InOutFusion", "Model"]
