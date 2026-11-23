from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mimose.models.abstract_model import AbstractModel

num_modals = 4
input_patch_size = 128

# External pipeline uses [t1c, t1n, t2f, t2w]; legacy ALoRA-MoE expects
# [T1, T1c, T2, FLAIR] (legacy/ALoRA-MoE/configs/brats2018.yaml).
DATASET_MODALITY_ORDER = (1, 0, 3, 2)


class LoRALinear(nn.Module):
    """Legacy ``LoRALinear``: a trainable linear layer plus a rank-``r`` update (``A`` ~ N(0, 0.01²), ``B`` = 0)."""

    def __init__(self, in_f: int, out_f: int, r: int = 8, alpha: float = 1.0, bias: bool = True, freeze_main: bool = False) -> None:
        super().__init__()
        self.main = nn.Linear(in_f, out_f, bias=bias)
        if freeze_main:
            for p in self.main.parameters():
                p.requires_grad = False
        self.A = nn.Parameter(torch.randn(out_f, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, in_f))
        self.scaling = alpha / max(1, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + (x @ self.B.t() @ self.A.t()) * self.scaling


class ExpertFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, r: int = 8) -> None:
        super().__init__()
        self.fc1 = LoRALinear(dim, hidden, r=r)
        self.act = nn.GELU()
        self.fc2 = LoRALinear(hidden, dim, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoELayer(nn.Module):
    """Legacy top-k MoE FFN: softmax over the top-k router logits, every expert run densely and masked."""

    def __init__(self, dim: int, hidden: int, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        # Legacy builds the experts with the default rank (8), not the model's lora_r.
        self.experts = nn.ModuleList([ExpertFFN(dim, hidden) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,T,dim)
        logits = self.router(x)
        topv, topi = logits.topk(self.top_k, dim=-1)
        weights = topv.softmax(dim=-1)
        # Legacy recomputes expert(x) for every top-k slot; each expert's output is computed
        # once here instead (identical result).
        expert_out: dict[int, torch.Tensor] = {}
        y = 0.0
        for j in range(self.top_k):
            idx = topi[..., j]
            out = 0.0
            for e, expert in enumerate(self.experts):
                mask = (idx == e).unsqueeze(-1)
                if mask.any():
                    if e not in expert_out:
                        expert_out[e] = expert(x)
                    out = out + torch.where(mask, expert_out[e], torch.zeros_like(x))
            y = y + weights[..., j].unsqueeze(-1) * out
        return y


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, r: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = LoRALinear(dim, dim, r=r)
        self.k = LoRALinear(dim, dim, r=r)
        self.v = LoRALinear(dim, dim, r=r)
        self.o = LoRALinear(dim, dim, r=r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,dim)
        h = self.heads
        q = rearrange(self.q(x), "b t (h c) -> b h t c", h=h)
        k = rearrange(self.k(x), "b t (h c) -> b h t c", h=h)
        v = rearrange(self.v(x), "b t (h c) -> b h t c", h=h)
        # softmax(q k^T * scale) v as in legacy, without materialising the T x T matrix
        # (32768 tokens at stride 4 of a 128^3 patch).
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.o(rearrange(out, "b h t c -> b t (h c)"))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, r: int = 8, num_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, r=r)
        self.norm2 = nn.LayerNorm(dim)
        self.moe = MoELayer(dim, dim * mlp_ratio, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch: int = 1, embed_dim: int = 64, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Encoder3D(nn.Module):
    """Legacy encoder: per-modality stride-4 patch embedding, four transformer stages joined by
    stride-2 convs. Returns the first three stage maps (strides 4, 8, 16; legacy names them
    s8/s16/s32). The fourth stage runs but its output is discarded, as in legacy."""

    def __init__(
        self,
        in_ch: int = 4,
        embed_dim: int = 64,
        depth: tuple[int, ...] = (1, 1, 2, 1),
        num_heads: tuple[int, ...] = (2, 4, 4, 8),
        r: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.patch = nn.ModuleList([PatchEmbed3D(1, embed_dim, patch_size=4) for _ in range(in_ch)])
        dims = [embed_dim * in_ch, embed_dim * in_ch * 2, embed_dim * in_ch * 4, embed_dim * in_ch * 8]
        self.proj = nn.ModuleList([nn.Conv3d(dims[i], dims[i + 1], 2, 2) for i in range(3)])
        self.blocks = nn.ModuleList()
        for stage, d in enumerate(depth):
            for _ in range(d):
                self.blocks.append(Block(dims[stage], num_heads[stage], r=r, num_experts=num_experts, top_k=top_k))
            if stage < 3:
                self.blocks.append(nn.Identity())  # stage delimiter
        self.dims = dims

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C = x.shape[:2]
        feats = [self.patch[i](x[:, i : i + 1]) for i in range(C)]
        z = torch.cat(feats, dim=1)
        outs = []
        t = z.flatten(2).transpose(1, 2)
        stage = 0
        for blk in self.blocks:
            if isinstance(blk, nn.Identity):
                outs.append(z)
                if stage < 3:
                    z = self.proj[stage](z)
                    t = z.flatten(2).transpose(1, 2)
                    stage += 1
                continue
            t = blk(t)
            z = t.transpose(1, 2).reshape(B, self.dims[stage], z.shape[2], z.shape[3], z.shape[4])
        outs.append(z)
        s8, s16, s32, _s64 = outs
        return s8, s16, s32


class HGDDecoder(nn.Module):
    """Legacy holistically guided decoder: codewords pooled at the coarsest scale, recombined with
    per-voxel codeword weights at the finest scale, then a 1x1 head."""

    def __init__(self, in_channels: tuple[int, int, int] = (256, 512, 1024), mid: int = 256, w: int = 32, out_ch: int = 4) -> None:
        super().__init__()
        c8, c16, c32 = in_channels
        self.c8 = nn.Conv3d(c8, 512, 1)
        self.c16 = nn.Conv3d(c16, 512, 1)
        self.c32 = nn.Conv3d(c32, 512, 1)

        self.bm = nn.Conv3d(512 * 3, mid, 1)  # base map
        self.sm = nn.Conv3d(512 * 3, w, 1)  # spatial weights

        self.g1 = nn.Conv3d(512 * 3, mid, 1)  # guidance
        self.g2 = nn.Conv3d(mid, w, 1)  # codeword weights

        self.head = nn.Conv3d(mid * 2, out_ch, 1)

    def _fuse(self, g8: torch.Tensor, g16: torch.Tensor, g32: torch.Tensor, to: str = "s32") -> torch.Tensor:
        if to == "s32":
            g8d = F.interpolate(g8, size=g32.shape[-3:], mode="trilinear", align_corners=False)
            g16d = F.interpolate(g16, size=g32.shape[-3:], mode="trilinear", align_corners=False)
            return torch.cat([g8d, g16d, g32], dim=1)
        g16u = F.interpolate(g16, size=g8.shape[-3:], mode="trilinear", align_corners=False)
        g32u = F.interpolate(g32, size=g8.shape[-3:], mode="trilinear", align_corners=False)
        return torch.cat([g8, g16u, g32u], dim=1)

    def forward(self, s8: torch.Tensor, s16: torch.Tensor, s32: torch.Tensor, out_size: tuple[int, int, int]) -> torch.Tensor:
        g8, g16, g32 = self.c8(s8), self.c16(s16), self.c32(s32)
        c32 = self._fuse(g8, g16, g32, to="s32")
        c8 = self._fuse(g8, g16, g32, to="s8")

        BM = self.bm(c32)
        SM = self.sm(c32)
        SM = SM.flatten(2).softmax(-1)  # spatial softmax
        BMf = BM.flatten(2).transpose(1, 2)
        CW = torch.einsum("bwn,bnq->bwq", SM, BMf)

        FVg = self.g1(c8)
        BMmean = BM.mean(dim=(2, 3, 4), keepdim=True)
        FVg = FVg + BMmean
        CWw = self.g2(FVg).flatten(2)
        CWw = CWw.softmax(1)
        FV8 = torch.einsum("bwq,bwk->bqk", CW, CWw).reshape(FVg.shape)
        Fcat = torch.cat([FV8, FVg], dim=1)
        logits_s8 = self.head(Fcat)
        # Legacy upsamples by a fixed x8, but its finest map is at stride 4, so the logits come out
        # at twice the input size and its training loss cannot run; resize to the input instead
        # (legacy patches.py expects logits "upsampled to input size").
        return F.interpolate(logits_s8, size=out_size, mode="trilinear", align_corners=False)


class ALoRAMoE(AbstractModel):
    """ALoRA-MoE (legacy/ALoRA-MoE, ``SegModel``): LoRA-adapted MoE transformer encoder with a
    holistically guided decoder. Missing modalities are zero-filled at the input; the network
    itself never sees the mask, as in legacy. Outputs 4 logit channels: channels 1..3 are
    trained as independent sigmoids and inference takes the argmax over all four, as legacy does.
    """

    def __init__(
        self,
        num_cls: int = 4,
        embed_dim: int = 64,
        depth: tuple[int, ...] | list[int] = (1, 1, 2, 1),
        num_heads: tuple[int, ...] | list[int] = (2, 4, 4, 8),
        lora_r: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        codewords: int = 32,
    ) -> None:
        super().__init__()
        self.num_cls = int(num_cls)
        self.enc = Encoder3D(
            in_ch=num_modals,
            embed_dim=embed_dim,
            depth=tuple(depth),
            num_heads=tuple(num_heads),
            r=lora_r,
            num_experts=num_experts,
            top_k=top_k,
        )
        c8 = num_modals * embed_dim
        c16 = num_modals * embed_dim * 2
        c32 = num_modals * embed_dim * 4
        # Legacy hard-codes w=32 in SegModel; its config's hgd.codewords is 32 too.
        self.dec = HGDDecoder(in_channels=(c8, c16, c32), mid=embed_dim * 4, w=codewords, out_ch=self.num_cls)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x, mask = self._remap_input_order(x, mask)
        x = x * mask.view(mask.size(0), num_modals, 1, 1, 1).to(x.dtype)
        s8, s16, s32 = self.enc(x)
        return self.dec(s8, s16, s32, out_size=tuple(x.shape[-3:]))

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Legacy ``sliding_window_inference``: 128^3 windows at 50% overlap, logits averaged
        uniformly. The last window of each axis is aligned to the volume edge so every voxel is
        covered (legacy's ``range`` stops short of the far edges)."""
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
                    window = (
                        slice(None),
                        slice(None),
                        slice(h, h + input_patch_size),
                        slice(w, w + input_patch_size),
                        slice(d, d + input_patch_size),
                    )
                    prediction[window] += self(images[window], mask)
                    weight[window] += 1
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
    def _remap_input_order(images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.size(1) != num_modals:
            raise RuntimeError(f"ALoRA-MoE expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"ALoRA-MoE expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")
        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = ALoRAMoE
