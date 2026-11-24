from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

num_modals = 4
input_patch_size = 128
bottleneck_size = 16  # input_patch_size // 8 (3 stride-2 downsamples)
embed_dim = 768
unetr_patch_size = 4
unetr_num_heads = 12
unetr_num_layers = 12
unetr_extract_layers = (3, 6, 9, 12)

# External pipeline uses [t1c, t1n, t2f, t2w]; MST-KDNet internals expect
# [flair, t1, t1ce, t2] (note: t1/t1ce order differs from the DCSeg/RFNet
# family's [flair, t1ce, t1, t2] convention).
DATASET_MODALITY_ORDER = (2, 1, 0, 3)


class BasicBlock(nn.Module):
    """Myronenko-style pre-activation residual block (GroupNorm->ReLU->Conv3d x2)."""

    def __init__(self, in_channels: int, out_channels: int, n_groups: int = 8) -> None:
        super().__init__()
        self.gn1 = nn.GroupNorm(n_groups, in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(n_groups, in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu1(self.gn1(x))
        x = self.conv1(x)
        x = self.relu2(self.gn2(x))
        x = self.conv2(x)
        return x + residual


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size, stride),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_size).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        weights = scores.softmax(dim=-1)
        attn = self.attn_drop(weights)

        context = (attn @ v).permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), context.size(1), -1)
        out = self.proj_drop(self.out(context))
        return out, weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.relu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = PositionwiseFeedForward(dim, 2048, dropout)
        self.attn = SelfAttention(num_heads, dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(self.attention_norm(x))
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, weights


class Embeddings(nn.Module):
    def __init__(self, input_dim: int, dim: int, patch_dim: tuple[int, int, int], patch_size: int, dropout: float) -> None:
        super().__init__()
        n_patches = patch_dim[0] * patch_dim[1] * patch_dim[2]
        self.patch_embeddings = nn.Conv3d(input_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class ViTEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim: int,
        patch_dim: tuple[int, int, int],
        patch_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        extract_layers: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.embeddings = Embeddings(input_dim, dim, patch_dim, patch_size, dropout)
        block = TransformerBlock(dim, num_heads, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(block) for _ in range(num_layers)])
        self.extract_layers = extract_layers

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        hidden_states = self.embeddings(x)
        extracted, extracted_weights = [], []
        for depth, layer in enumerate(self.layers):
            hidden_states, weights = layer(hidden_states)
            if depth + 1 in self.extract_layers:
                extracted.append(hidden_states)
                extracted_weights.append(weights)
        return extracted, extracted_weights


class UNETR(nn.Module):
    """3D UNETR bottleneck transformer producing a residual refinement plus
    intermediate deconv features/attention maps used for distillation."""

    def __init__(
        self,
        bottleneck_shape: tuple[int, int, int],
        input_dim: int = 128,
        output_dim: int = 128,
        patch_size: int = unetr_patch_size,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_dim = tuple(dim // patch_size for dim in bottleneck_shape)

        self.transformer = ViTEncoder(
            input_dim,
            embed_dim,
            self.patch_dim,
            patch_size,
            unetr_num_heads,
            unetr_num_layers,
            dropout=0.1,
            extract_layers=unetr_extract_layers,
        )

        self.decoder0 = nn.Sequential(Conv3DBlock(input_dim, 32), Conv3DBlock(32, 64))
        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim, 512), Deconv3DBlock(512, 256), Deconv3DBlock(256, 128)
        )
        self.decoder6 = nn.Sequential(Deconv3DBlock(embed_dim, 512), Deconv3DBlock(512, 256))
        self.decoder9 = Deconv3DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512), Conv3DBlock(512, 512), Conv3DBlock(512, 512), SingleDeconv3DBlock(512, 256)
        )
        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256), Conv3DBlock(256, 256), SingleDeconv3DBlock(256, 128)
        )
        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128, stride=2), Conv3DBlock(128, 128, stride=2), SingleDeconv3DBlock(128, 64)
        )
        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64), Conv3DBlock(64, 64), SingleConv3DBlock(64, output_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        extracted, extract_weights = self.transformer(x)
        z3, z6, z9, z12 = extracted

        def _to_grid(tokens: torch.Tensor) -> torch.Tensor:
            return tokens.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z3, z6, z9, z12 = _to_grid(z3), _to_grid(z6), _to_grid(z9), _to_grid(z12)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(x)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output, [z3, z6, z9, z12], extract_weights


class _MSTKDNetBackbone(nn.Module):
    """Myronenko 3D U-Net with a UNETR bottleneck transformer.

    Returns the full legacy 7-tuple ``(uout, style, content, unetr_fs,
    extract_weights, Global_style, logit)`` so a single class can serve as
    both the "full" (teacher) and "missing" (student) co-training branch.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, init_channels: int = 16) -> None:
        super().__init__()
        self.conv1a = nn.Conv3d(in_channels, init_channels, kernel_size=3, padding=1)
        self.conv1b = BasicBlock(init_channels, init_channels)
        self.ds1 = nn.Conv3d(init_channels, init_channels * 2, kernel_size=3, stride=2, padding=1)

        self.conv2a = BasicBlock(init_channels * 2, init_channels * 2)
        self.conv2b = BasicBlock(init_channels * 2, init_channels * 2)
        self.ds2 = nn.Conv3d(init_channels * 2, init_channels * 4, kernel_size=3, stride=2, padding=1)

        self.conv3a = BasicBlock(init_channels * 4, init_channels * 4)
        self.conv3b = BasicBlock(init_channels * 4, init_channels * 4)
        self.ds3 = nn.Conv3d(init_channels * 4, init_channels * 8, kernel_size=3, stride=2, padding=1)

        self.conv4a = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4b = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4c = BasicBlock(init_channels * 8, init_channels * 8)
        self.conv4d = BasicBlock(init_channels * 8, init_channels * 8)

        bottleneck_shape = (bottleneck_size, bottleneck_size, bottleneck_size)
        self.unetr = UNETR(bottleneck_shape, input_dim=init_channels * 8, output_dim=init_channels * 8)

        self.up4conva = nn.Conv3d(init_channels * 8, init_channels * 4, kernel_size=1)
        self.up4 = nn.Upsample(scale_factor=2)
        self.up4convb = BasicBlock(init_channels * 4, init_channels * 4)

        self.up3conva = nn.Conv3d(init_channels * 4, init_channels * 2, kernel_size=1)
        self.up3 = nn.Upsample(scale_factor=2)
        self.up3convb = BasicBlock(init_channels * 2, init_channels * 2)

        self.up2conva = nn.Conv3d(init_channels * 2, init_channels, kernel_size=1)
        self.up2 = nn.Upsample(scale_factor=2)
        self.up2convb = BasicBlock(init_channels, init_channels)

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.convc = nn.Conv3d(init_channels * 20, init_channels * 8, kernel_size=1)
        self.convco = nn.Conv3d(init_channels * 16, init_channels * 8, kernel_size=1)
        self.up1conv = nn.Conv3d(init_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        c1 = self.conv1a(x)
        c1 = self.conv1b(c1)
        c1d = self.ds1(c1)

        c2 = self.conv2a(c1d)
        c2 = self.conv2b(c2)
        c2d = self.ds2(c2)
        c2d_p = self.pool(c2d)

        c3 = self.conv3a(c2d)
        c3 = self.conv3b(c3)
        c3d = self.ds3(c3)

        output, unetr_fs, extract_weights = self.unetr(c3d)
        output = output + c3d

        c4 = self.conv4a(output)
        c4 = self.conv4b(c4)
        c4 = self.conv4c(c4)
        c4d = self.conv4d(c4)

        style = self.convc(torch.cat([c2d_p, c3d, output], dim=1))
        content = c4d
        global_style = [c3d, style, c4d]

        c4d = self.convco(torch.cat([style, content], dim=1))
        c4d = self.dropout(c4d)

        u4 = self.up4conva(c4d)
        u4 = self.up4(u4)
        u4 = u4 + c3
        u4 = self.up4convb(u4)

        u3 = self.up3conva(u4)
        u3 = self.up3(u3)
        u3 = u3 + c2
        u3 = self.up3convb(u3)

        u2 = self.up2conva(u3)
        u2 = self.up2(u2)
        u2 = u2 + c1
        u2 = self.up2convb(u2)

        logit = self.up1conv(u2)
        uout = torch.sigmoid(logit)

        return uout, style, content, unetr_fs, extract_weights, global_style, logit


class MSTKDNet(AbstractModel):
    """Implementation of MST-KDNet introduced in [1].

    Legacy trains two full-weight instances of the same backbone jointly: a
    "full" branch (teacher, always sees the true unmasked 4-channel volume)
    and a "missing" branch (student, sees the volume zero-filled per the
    modality mask), with multiple distillation losses (bottleneck content
    MSE, UNETR intermediate-feature MSE, attention extreme-value distillation,
    logit-standardized KL, and a Gram-matrix-based global style match) pulling
    the student toward the teacher. Only the student branch is ever evaluated
    at inference (confirmed via legacy's own `eval.py`), so this port keeps
    both branches as submodules of one model (mirroring the teacher/student
    contract) but ``predict()``/inference-mode ``forward()`` only exercises
    the student.

    Legacy additionally trains a small PatchGAN-style discriminator to align
    the student's "style" feature with the teacher's, via a second Adam
    optimizer that is (seemingly by mistake) reconstructed from scratch every
    training iteration, discarding its momentum state each step. Given that
    bug, the discriminator's minor loss weight (`2e-4`), and the added
    complexity of a second optimizer in a single-optimizer trainer contract,
    this port drops the discriminator entirely — a training-time-only,
    secondary auxiliary, not the paper's central contribution.

    Legacy hardcodes `mask.view(1, 4, 1, 1, 1)` when building the missing
    branch's zero-filled input, which silently assumes `batch_size == 1`
    (always true in legacy's own configs) — this port uses the actual batch
    size instead so this works for any batch size.

    [1] original MST-KDNet missing-modality brain tumor segmentation model.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.student = _MSTKDNetBackbone(in_channels=num_modals, out_channels=num_cls)
        self.teacher = _MSTKDNetBackbone(in_channels=num_modals, out_channels=num_cls)
        self.is_training = False

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[tuple, tuple]:
        x, mask = self._remap_input_order(x, mask)
        masked = x * mask.view(mask.size(0), num_modals, 1, 1, 1).to(x.dtype)

        student_out = self.student(masked)
        if not self.is_training:
            return student_out[0]

        teacher_out = self.teacher(x)
        return student_out, teacher_out

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0), self.student.up1conv.out_channels, height, width, depth, device=images.device
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
            raise RuntimeError(f"MSTKDNet expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"MSTKDNet expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = MSTKDNet


__all__ = ["MSTKDNet", "Model"]
