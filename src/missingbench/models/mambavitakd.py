from __future__ import annotations

import torch
import torch.nn as nn

from missingbench.models.abstract_model import AbstractModel

basic_dims = 16
num_modals = 4
input_patch_size = 80

# External pipeline uses [t1c, t1n, t2f, t2w]; legacy MambaVit-AKD's own
# BraTS loader (and the model's own internal indexing) stacks modalities as
# [flair, t1ce, t1, t2], the same convention already used by the
# RFNet/A2FSeg/IMFuse family in this codebase.
DATASET_MODALITY_ORDER = (2, 0, 1, 3)

MambaVitAKDBranchOutput = tuple[
    torch.Tensor,  # fuse_pred (softmax probabilities)
    tuple[torch.Tensor, ...],  # sep_preds
    tuple[torch.Tensor, ...],  # prm_preds
    torch.Tensor,  # feature (pre-logit fused decoder feature, "de_x1")
    torch.Tensor,  # logits (pre-softmax fused logits)
]
MambaVitAKDOutput = (
    torch.Tensor | tuple[MambaVitAKDBranchOutput, MambaVitAKDBranchOutput, torch.Tensor]
)


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
        relufactor: float = 0.2,
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
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            raise ValueError(f"activation type {act_type} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class PRMGeneratorLastStage(nn.Module):
    def __init__(
        self, in_channel: int = 64, num_cls: int = 4, num_modal: int = 4
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(
                in_channel * num_modal, in_channel // 4, k_size=1, padding=0, stride=1
            ),
            GeneralConv3d(
                in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1
            ),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x.size()
        masked = torch.zeros_like(x)
        masked[mask, ...] = x[mask, ...]
        masked = masked.view(batch_size, -1, height, width, depth)
        return self.prm_layer(self.embedding_layer(masked))


class PRMGenerator(nn.Module):
    def __init__(
        self, in_channel: int = 64, num_cls: int = 4, num_modal: int = 4
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(
                in_channel * num_modal, in_channel // 4, k_size=1, padding=0, stride=1
            ),
            GeneralConv3d(
                in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1
            ),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x2.size()
        masked = torch.zeros_like(x2)
        masked[mask, ...] = x2[mask, ...]
        masked = masked.view(batch_size, -1, height, width, depth)
        return self.prm_layer(torch.cat((x1, self.embedding_layer(masked)), dim=1))


class ModalFusion(nn.Module):
    def __init__(self, in_channel: int = 64, num_modal: int = 4) -> None:
        super().__init__()
        # Legacy MambaVit-AKD's own layers.py uses a 96-unit hidden layer here
        # (unlike this codebase's RFNet/IMFuse/etc. ports, which all use 128).
        self.weight_layer = nn.Sequential(
            nn.Conv3d(num_modal * in_channel + 1, 96, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(96, num_modal, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, prm: torch.Tensor) -> torch.Tensor:
        batch_size, modalities, channels, _, _, _ = x.size()
        prm_avg = torch.mean(prm, dim=(3, 4, 5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg
        feat_avg = feat_avg.view(batch_size, modalities * channels, 1, 1, 1)
        feat_avg = torch.cat(
            (feat_avg, prm_avg[:, 0, 0, ...].view(batch_size, 1, 1, 1, 1)),
            dim=1,
        )
        weight = self.weight_layer(feat_avg).view(batch_size, modalities, 1)
        weight = self.sigmoid(weight).view(batch_size, modalities, 1, 1, 1, 1)
        return torch.sum(x * weight, dim=1)


class RegionFusion(nn.Module):
    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3d(
                in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1
            ),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x.size()
        x = x.view(batch_size, -1, height, width, depth)
        return self.fusion_layer(x)


class RegionAwareModalFusion(nn.Module):
    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.num_cls = num_cls
        self.modal_fusion = nn.ModuleList(
            [ModalFusion(in_channel=in_channel) for _ in range(num_cls)]
        )
        self.region_fusion = RegionFusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
            GeneralConv3d(
                in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1
            ),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1),
        )

    def forward(
        self, x: torch.Tensor, prm: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x.size()
        masked = torch.zeros_like(x)
        masked[mask, ...] = x[mask, ...]

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, x.size(2), 1, 1, 1)
        flair = masked[:, 0:1, ...] * prm
        t1ce = masked[:, 1:2, ...] * prm
        t1 = masked[:, 2:3, ...] * prm
        t2 = masked[:, 3:4, ...] * prm
        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, index, :, :] for index in range(self.num_cls)]

        region_fused_feat = []
        for index in range(self.num_cls):
            region_fused_feat.append(
                self.modal_fusion[index](
                    region_feat[index], prm[:, index : index + 1, ...]
                )
            )
        region_fused_feat = torch.stack(region_fused_feat, dim=1)
        return torch.cat(
            (
                self.region_fusion(region_fused_feat),
                self.short_cut(masked.view(batch_size, -1, height, width, depth)),
            ),
            dim=1,
        )


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.e1_c1 = GeneralConv3d(1, basic_dims, pad_type="reflect")
        self.e1_c2 = GeneralConv3d(basic_dims, basic_dims, pad_type="reflect")
        self.e1_c3 = GeneralConv3d(basic_dims, basic_dims, pad_type="reflect")

        self.e2_c1 = GeneralConv3d(
            basic_dims, basic_dims * 2, stride=2, pad_type="reflect"
        )
        self.e2_c2 = GeneralConv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = GeneralConv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")

        self.e3_c1 = GeneralConv3d(
            basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect"
        )
        self.e3_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = GeneralConv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")

        self.e4_c1 = GeneralConv3d(
            basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect"
        )
        self.e4_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = GeneralConv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")

    def forward(
        self, x: torch.Tensor
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

        self.seg_layer = nn.Conv3d(
            basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        de_x4 = self.d3_c1(self.d3(x4))
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x4, x3), dim=1)))
        de_x3 = self.d2_c1(self.d2(de_x3))
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x3, x2), dim=1)))
        de_x2 = self.d1_c1(self.d1(de_x2))
        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x2, x1), dim=1)))
        return self.softmax(self.seg_layer(de_x1))


class DecoderFuse(nn.Module):
    """Region-aware modal fusion decoder (legacy ``Decoder_fuse``).

    Identical RFM/PRM architecture to ``rfnet.py``'s ``DecoderFuse``, except
    it additionally returns the pre-logit fused feature (``de_x1``) and the
    pre-softmax logits alongside the softmax prediction and PRM predictions
    -- these two extra outputs are what legacy's knowledge-distillation
    losses (KD/prototype/attention, see ``MambaVitAKDLoss``) are computed
    from.
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

        self.seg_layer = nn.Conv3d(
            basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)

        self.RFM4 = RegionAwareModalFusion(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = RegionAwareModalFusion(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = RegionAwareModalFusion(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = RegionAwareModalFusion(in_channel=basic_dims, num_cls=num_cls)

        self.prm_generator4 = PRMGeneratorLastStage(
            in_channel=basic_dims * 8, num_cls=num_cls
        )
        self.prm_generator3 = PRMGenerator(in_channel=basic_dims * 4, num_cls=num_cls)
        self.prm_generator2 = PRMGenerator(in_channel=basic_dims * 2, num_cls=num_cls)
        self.prm_generator1 = PRMGenerator(in_channel=basic_dims, num_cls=num_cls)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = self.d3_out(self.d3_c2(torch.cat((de_x3, de_x4), dim=1)))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = self.d2_out(self.d2_c2(torch.cat((de_x2, de_x3), dim=1)))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = self.d1_out(self.d1_c2(torch.cat((de_x1, de_x2), dim=1)))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        return (
            de_x1,
            logits,
            pred,
            (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4)),
        )


class ChannelAttention(nn.Module):
    """Legacy ``ChannelAttention`` (SE-style avg+max pool channel gate).

    Legacy instantiates a *new*, randomly-initialized ``ChannelAttention``
    inside ``attention_loss`` on every training step, so its weights are
    never actually optimized (an apparent bug -- the module has learnable
    parameters but they are discarded after each call and never receive a
    gradient update across steps). This port keeps a single persistent
    instance as a submodule instead, so it is optimized like every other
    parameter, while preserving the mechanism (a channel-attention-gated MSE
    between student/teacher features).
    """

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _, _ = x.size()
        avg_out = self.fc2(self.fc1(self.avg_pool(x).view(batch_size, channels)))
        max_out = self.fc2(self.fc1(self.max_pool(x).view(batch_size, channels)))
        attention_weights = self.sigmoid(avg_out + max_out).view(
            batch_size, channels, 1, 1, 1
        )
        return x * attention_weights


class _Branch(nn.Module):
    """One full encoder/decoder branch (4 per-modality encoders + the fused
    and per-modality decoders) -- legacy's ``model_RFNet.Model``, reused
    verbatim as both the teacher and (with the Mamba encoder dropped, see
    ``MambaVitAKD`` docstring) the student branch."""

    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.decoder_fuse = DecoderFuse(num_cls=num_cls)
        self.decoder_sep = DecoderSep(num_cls=num_cls)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        with_sep: bool,
    ) -> MambaVitAKDBranchOutput:
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1)
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        feature, logits, fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        sep_preds: tuple[torch.Tensor, ...] = ()
        if with_sep:
            sep_preds = (
                self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4),
                self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4),
                self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4),
                self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4),
            )
        return fuse_pred, sep_preds, prm_preds, feature, logits


class MambaVitAKD(AbstractModel):
    """Port of legacy MambaVit-AKD (repo folder name; the live model class is
    ``model_mamba_modalitys.Model``, trained against a separately-pretrained,
    frozen ``model_RFNet.Model`` teacher via adversarial-prototype knowledge
    distillation, or "apkd").

    Architecture notes:
    - Legacy's student decoder architecture (encoders + region-aware-fusion
      ``Decoder_fuse`` + shared ``Decoder_sep``) is *exactly* RFNet's
      architecture (see ``rfnet.py``), reused verbatim here as ``_Branch``.
      Upstream (xixiaa009/MambaVit-AKD @ e85fa89) additionally stacks the
      output of a SegMamba encoder as a fifth "modality", but that encoder
      is rebuilt with fresh random weights inside every ``forward`` call
      (never registered or optimised), and the released code does not run
      (NameError in ``segMamba.py``, undefined ``model_baseline.Model``,
      4-modality hard-coding in ``layers.py``); its ViT ``Encoder0`` modules
      are built but unused. This port therefore drops the Mamba and ViT
      branches: what remains is the trainable part of the method, RFNet plus
      the three distillation terms.
    - The "teacher" is architecturally identical to the student (also
      ``_Branch``, i.e. RFNet), always run on the full 4-modality input.
      Legacy pretrains this teacher in a wholly separate training run
      (``train_RFNet.py``) and loads it frozen from a checkpoint path before
      distilling into the student. MissingBench's trainer contract has no notion
      of loading a second, externally-pretrained model (only the same
      model's own checkpoints can be resumed/pretrained-from), so -- mirroring
      how this codebase's other multi-network distillation model
      (``MSTKDNet``) was already adapted -- the teacher is embedded as a
      second submodule and trained jointly with the student in a single run
      instead: it receives its own direct segmentation loss on full
      modalities (see ``MambaVitAKDLoss``), and its outputs are detached
      before being used as the distillation target, so the KD/prototype/
      attention terms never backpropagate into the teacher (matching
      legacy's ``torch.no_grad()`` teacher forward pass), while the
      teacher's own weights are still optimized via its own segmentation
      loss instead of staying frozen at a pretrained checkpoint.
    - At inference, only the student branch (run with the true, possibly
      partial, modality mask) is ever evaluated -- confirmed via legacy's
      own ``predict.py``/``trainer.py``, which validate exclusively with
      ``model`` (the Mamba/student model), never ``teacher_model``.
    """

    def __init__(
        self,
        num_cls: int = 4,
        patch_size: int = input_patch_size,
        trainable_attention_gate: bool = False,
        teacher_random_masks: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        # Upstream train_RFNet.py pretrains the teacher on random modality subsets, not the full set.
        self.teacher_random_masks = teacher_random_masks
        # Set by the trainer once distillation starts: the teacher then runs under no_grad without
        # its per-modality decoders (upstream loads it frozen from a separate run).
        self.teacher_frozen = False
        self.student = _Branch(num_cls=num_cls)
        self.teacher = _Branch(num_cls=num_cls)
        # Channel-attention gate used by the attention-distillation loss term
        # (see ``ChannelAttention`` docstring for why this is a persistent
        # submodule rather than legacy's per-call throwaway instance). Its
        # input channel count is ``basic_dims`` -- the fused decoder feature
        # ("de_x1") both branches emit.
        self.channel_attention = ChannelAttention(in_channels=basic_dims)
        # Upstream builds a fresh random gate on every call, so it is never trained; a trainable
        # shared gate could shrink the attention loss by pushing itself toward 0.
        self.channel_attention.requires_grad_(trainable_attention_gate)
        self.is_training = False

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        mode: str = "auto",
    ) -> MambaVitAKDOutput:
        """``mode="auto"`` (default) is the normal student(+teacher) path used
        by inference and joint distillation training, gated by
        ``self.is_training``. ``mode="teacher_pretrain"`` instead runs *only*
        the teacher branch (always on the full-modality mask, ``with_sep=True``),
        for ``MambaVitAKDTrainer``'s dedicated teacher-pretraining phase --
        this keeps the call going through ``self.model(...)`` rather than
        ``self.model.teacher(...)`` directly, so DistributedDataParallel's
        gradient-sync hooks stay attached (see other models' ``forward``
        docstrings for the same DDP-safety concern)."""
        x, mask = self._remap_input_order(x, mask)

        if mode == "teacher_pretrain":
            teacher_mask = mask if self.teacher_random_masks else torch.ones_like(mask)
            return self.teacher(x, teacher_mask, with_sep=True)

        student_out = self.student(x, mask, with_sep=self.is_training)
        if not self.is_training:
            return student_out[0]

        full_mask = torch.ones_like(mask)
        if self.teacher_frozen:
            with torch.no_grad():
                teacher_out = self.teacher(x, full_mask, with_sep=False)
        else:
            teacher_out = self.teacher(x, full_mask, with_sep=True)

        feature, feature_t = student_out[3], teacher_out[3]
        attn_loss = nn.functional.mse_loss(
            self.channel_attention(feature),
            self.channel_attention(feature_t.detach()),
        )
        return student_out, teacher_out, attn_loss

    def _sliding_window_predict(
        self, images: torch.Tensor, patch_fn
    ) -> torch.Tensor:
        """Shared 50%-overlap sliding-window inference loop used by both
        ``predict`` (student) and ``predict_teacher`` -- ``patch_fn`` maps one
        ``patch_size``-sized crop to its softmax prediction."""
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (
            self.patch_size,
            self.patch_size,
            self.patch_size,
        ):
            return patch_fn(images)

        h_starts = self._window_starts(height, self.patch_size)
        w_starts = self._window_starts(width, self.patch_size)
        d_starts = self._window_starts(depth, self.patch_size)
        prediction = torch.zeros(
            images.size(0),
            self.student.decoder_fuse.seg_layer.out_channels,
            height,
            width,
            depth,
            device=images.device,
        )
        weight = torch.zeros(
            images.size(0), 1, height, width, depth, device=images.device
        )

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + self.patch_size,
                        w : w + self.patch_size,
                        d : d + self.patch_size,
                    ]
                    patch_pred = patch_fn(patch)
                    prediction[
                        :,
                        :,
                        h : h + self.patch_size,
                        w : w + self.patch_size,
                        d : d + self.patch_size,
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + self.patch_size,
                        w : w + self.patch_size,
                        d : d + self.patch_size,
                    ] += 1
        return prediction / weight

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._sliding_window_predict(images, lambda patch: self(patch, mask))

    def predict_teacher(self, images: torch.Tensor) -> torch.Tensor:
        """Sliding-window inference through the teacher branch alone (always
        full-modality), for validating ``MambaVitAKDTrainer``'s dedicated
        teacher-pretraining phase before the student/distillation phase
        starts."""
        full_mask = torch.ones(images.size(0), num_modals, dtype=torch.bool, device=images.device)

        def patch_fn(patch: torch.Tensor) -> torch.Tensor:
            fuse_pred_t, _, _, _, _ = self(patch, full_mask, mode="teacher_pretrain")
            return fuse_pred_t

        return self._sliding_window_predict(images, patch_fn)

    @staticmethod
    def _window_starts(size: int, patch_size: int = input_patch_size) -> list[int]:
        if size < patch_size:
            raise RuntimeError(
                "Model prediction expects spatial dimensions to be at least "
                f"{patch_size}, got {size}"
            )
        if size == patch_size:
            return [0]

        stride = patch_size // 2
        starts = list(range(0, max(size - patch_size, 0), stride))
        last_start = size - patch_size
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
                f"MambaVitAKD expects {num_modals} input modalities, got "
                f"{images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(
                f"MambaVitAKD expects mask shape [B, {num_modals}], got "
                f"{tuple(mask.shape)}"
            )

        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = MambaVitAKD


__all__ = ["MambaVitAKD", "Model"]
