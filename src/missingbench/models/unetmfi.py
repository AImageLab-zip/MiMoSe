from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel

num_modals = 4
input_patch_size = 128

# External pipeline uses [t1c, t1n, t2f, t2w]; UNET-MFI internals expect
# [flair, t1ce, t1, t2] (same convention as DCSeg/RFNet/M2FTrans/IMFuse) --
# confirmed by legacy `Ddataset.py`'s comment `x1,x2,x3,x4 = ... # ('flair',
# 't1ce', 't1', 't2')`.
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


class MaskModal(nn.Module):
    """Zero out raw voxel values for modalities marked absent, per sample.

    Legacy UNET-MFI's dataset zeroes each modality's raw input
    (`x1 * mask_code[0]`, etc.) before the model ever sees it, once per
    sample. MissingBench's dataset instead hands the model the full raw tensor
    plus a separate `[B, 4]` mask, so this module reproduces the same
    zero-fill inside the model, per sample (broadcasting `mask[:, i]` over
    the channel/spatial dims of stream `i`), not just for the first sample
    in the batch.
    """

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``x`` is a single modality's raw slice ``[B, 1, H, W, D]``; ``mask``
        is that modality's per-sample availability, shape ``[B]``."""
        return x * mask.view(-1, 1, 1, 1, 1).float()


class MFIBlock(nn.Module):
    """Modality-adaptive Feature Interaction block (the paper's core idea).

    For each of the 4 output branches, a small MLP ("Relation" network)
    consumes the pairwise relation between every source modality's globally
    pooled feature and the branch's own anchor-modality pooled feature,
    concatenated with the modality-availability code, and predicts a
    per-channel softmax weight over the 4 modalities. Each branch's enhanced
    feature is the weighted combination of all 4 raw modality feature maps
    (per channel), added residually to that branch's own feature.

    Legacy (`Model.py`'s `fianl_diff_code_block`) constructs 4 independent
    `Relation1..Relation4` submodules but its `forward` calls `self.Relation1`
    for all 4 branches -- `Relation2/3/4` are constructed and never used in
    the forward graph (dead weights). This port fixes that and actually uses
    one independent relation network per branch, as the 4 separate
    constructors evidently intended.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        def _make_relation() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(channels * 4, channels * 2),
                nn.LeakyReLU(),
                nn.Linear(channels * 2, channels),
            )

        self.relations = nn.ModuleList([_make_relation() for _ in range(num_modals)])

    def forward(
        self,
        feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        mod_code: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, channels, height, width, depth = feats[0].shape
        x_ori = torch.stack(feats, dim=1)  # [B, 4, C, H, W, D]
        pooled = x_ori.reshape(batch, num_modals, channels, -1).mean(-1)  # [B, 4, C]

        x1 = pooled.unsqueeze(1).repeat(1, num_modals, 1, 1)  # x1[:, i, j] = pooled[:, j]
        x2 = pooled.unsqueeze(2).repeat(1, 1, num_modals, 1)  # x2[:, i, j] = pooled[:, i]
        relation_pairs = torch.cat((x1, x2), dim=-1)  # [B, 4, 4, 2C]

        mod_code_exp = mod_code.unsqueeze(-1).repeat(1, 1, 2 * channels).float()  # [B, 4, 2C]

        outputs = []
        for branch_index in range(num_modals):
            relation_input = torch.cat((relation_pairs[:, branch_index, :, :], mod_code_exp), dim=-1)
            weights = self.relations[branch_index](relation_input)  # [B, 4, C]
            weights = F.softmax(weights, dim=1)
            combo = (x_ori * weights.view(batch, num_modals, channels, 1, 1, 1)).sum(dim=1)
            outputs.append(feats[branch_index] + combo)

        return tuple(outputs)


class EnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(in_channels // 8, in_channels),
            nn.ReLU(True),
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(in_channels // 8, in_channels),
            nn.ReLU(True),
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class EnDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DnUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.ConvTranspose3d(in_channels, out_channels, 2, 2),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + skip


class ModalityStream(nn.Module):
    """One modality's private encoder + decoder tower (no weight sharing).

    Mirrors legacy's `no_share_unet`, which builds 4 completely independent
    encoder/decoder stacks (`*_1`, `*_2`, `*_3`, `*_4`) -- one per modality --
    rather than a single shared-weight encoder run 4 times.
    """

    def __init__(self, num_cls: int) -> None:
        super().__init__()
        self.init_conv = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.en1 = EnBlock(32)
        self.ed1 = EnDown(32, 64)
        self.en2 = EnBlock(64)
        self.ed2 = EnDown(64, 128)
        self.en3 = EnBlock(128)
        self.ed3 = EnDown(128, 256)
        self.en4 = EnBlock(256)

        self.ud1 = DnUp(256, 128)
        self.un1 = EnBlock(128)
        self.ud2 = DnUp(128, 64)
        self.un2 = EnBlock(64)
        self.ud3 = DnUp(64, 32)
        self.un3 = EnBlock(32)

        self.out_conv = nn.Conv3d(32, num_cls, 1)
        self.stage1_out = nn.Sequential(
            nn.Conv3d(128, num_cls, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True),
        )
        self.stage2_out = nn.Sequential(
            nn.Conv3d(64, num_cls, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
        )

    def forward_encoder_stage1(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.init_conv(x)
        c1 = self.en1(x)
        p1 = self.ed1(c1)
        return c1, p1

    def forward_encoder_stage2(self, p1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c2 = self.en2(p1)
        p2 = self.ed2(c2)
        return c2, p2

    def forward_encoder_stage3(self, p2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c3 = self.en3(p2)
        p3 = self.ed3(c3)
        return c3, p3

    def forward_bottleneck(self, p3: torch.Tensor) -> torch.Tensor:
        return self.en4(p3)

    def forward_decoder_stage1(self, c4: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
        return self.ud1(c4, c3)

    def forward_decoder_stage2(self, un5: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        return self.ud2(un5, c2)

    def forward_decoder_stage3(self, un6: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
        return self.ud3(un6, c1)


class UNETMFI(AbstractModel):
    """Implementation of UNET-MFI introduced in [1].

    [1] Zhang, et al. "Modality-adaptive Feature Interaction for Brain Tumor
        Segmentation with Missing Modalities". MICCAI 2022.

    Architecture: 4 fully independent per-modality 3D U-Net towers (no
    shared weights, unlike mmformer/M2FTrans-style siamese encoders). The
    distinguishing idea is the Modality-adaptive Feature Interaction (MFI)
    block (`MFIBlock`): at 3 encoder resolutions (64/128/256 channels) and 2
    decoder resolutions (128/64 channels), every stream's feature map is
    updated with a modality-availability-conditioned, per-channel softmax
    combination of all 4 streams' globally-pooled features, added
    residually. There is no cross-modal attention or shared bottleneck --
    each modality's tower stays private except for these 5 MFI injections.

    Deep supervision produces per-modality predictions from two intermediate
    decoder stages (`stage1_out`/`stage2_out`, each upsampled internally to
    full resolution) plus each stream's own final output, and a final "fuse"
    prediction obtained from a small conv fusing the 4 streams' final
    per-modality logits. This maps directly onto MissingBench's IMFuse-style
    training contract: `fuse_pred` = the fused conv output, `sep_preds` = the
    4 per-modality final outputs, `prm_preds` = the 8 upsampled deep
    supervision outputs (4 modalities x 2 stages) -- so this port reuses
    `IMFuseTrainer`/`IMFuseLoss` verbatim rather than writing new ones.

    Legacy zeroes each modality's raw input once per sample inside the
    dataset (`x1 * mask_code[0]`, ...); this port instead applies the same
    zero-fill inside the model via `MaskModal`, per sample, since MissingBench's
    dataset hands the model the full raw tensor plus a separate mask.

    Legacy also outputs 3 sigmoid BraTS-region channels (WT/TC/ET) trained
    with `BCEDiceLoss`, rather than a softmax `num_cls`-channel prediction.
    Like every other model in this framework, this port instead uses a
    standard softmax `num_cls`-channel head (reusing
    `softmax_weighted_loss`/`dice_loss` from `losses/imfuse.py`) so it is
    compatible with `BaseTrainer`'s shared testing pipeline and generalizes
    to BraTS25's 5 classes -- legacy's `class_nums = out_channel` was always
    hardcoded to 3 in `train.py`, which would silently break for a
    non-region-based, non-3-class target.
    """

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.masker = MaskModal()

        self.streams = nn.ModuleList([ModalityStream(num_cls) for _ in range(num_modals)])

        self.diff1 = MFIBlock(64)
        self.diff2 = MFIBlock(128)
        self.diff3 = MFIBlock(256)
        self.diff4 = MFIBlock(128)
        self.diff5 = MFIBlock(64)

        self.cat_out_conv = nn.Conv3d(4 * num_cls, num_cls, 1)
        self.softmax = nn.Softmax(dim=1)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        x, mask = self._remap_input_order(x, mask)
        mod_code = mask.float()

        streams = self.streams
        inputs = [self.masker(x[:, i : i + 1], mask[:, i]) for i in range(num_modals)]

        c1s, p1s = zip(*(streams[i].forward_encoder_stage1(inputs[i]) for i in range(num_modals)))
        p1s = self.diff1(p1s, mod_code)

        c2s, p2s = zip(*(streams[i].forward_encoder_stage2(p1s[i]) for i in range(num_modals)))
        p2s = self.diff2(p2s, mod_code)

        c3s, p3s = zip(*(streams[i].forward_encoder_stage3(p2s[i]) for i in range(num_modals)))
        p3s = self.diff3(p3s, mod_code)

        c4s = [streams[i].forward_bottleneck(p3s[i]) for i in range(num_modals)]

        up5s = [streams[i].forward_decoder_stage1(c4s[i], c3s[i]) for i in range(num_modals)]
        stage1_preds = [self.softmax(streams[i].stage1_out(up5s[i])) for i in range(num_modals)]

        un5s = tuple(streams[i].un1(up5s[i]) for i in range(num_modals))
        un5s = self.diff4(un5s, mod_code)

        up6s = [streams[i].forward_decoder_stage2(un5s[i], c2s[i]) for i in range(num_modals)]
        stage2_preds = [self.softmax(streams[i].stage2_out(up6s[i])) for i in range(num_modals)]

        un6s = tuple(streams[i].un2(up6s[i]) for i in range(num_modals))
        un6s = self.diff5(un6s, mod_code)

        up7s = [streams[i].forward_decoder_stage3(un6s[i], c1s[i]) for i in range(num_modals)]
        un7s = [streams[i].un3(up7s[i]) for i in range(num_modals)]

        logits = [streams[i].out_conv(un7s[i]) for i in range(num_modals)]
        sep_preds = tuple(self.softmax(logit) for logit in logits)

        cat_logits = self.cat_out_conv(torch.cat(logits, dim=1))
        fuse_pred = self.softmax(cat_logits)

        if not self.is_training:
            return fuse_pred

        prm_preds = tuple(stage1_preds) + tuple(stage2_preds)
        return fuse_pred, sep_preds, prm_preds

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.num_cls,
            height,
            width,
            depth,
            device=images.device,
        )
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
            raise RuntimeError(f"UNETMFI expects {num_modals} input modalities, got {images.size(1)}")
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(f"UNETMFI expects mask shape [B, {num_modals}], got {tuple(mask.shape)}")

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = UNETMFI


__all__ = ["UNETMFI", "Model"]
