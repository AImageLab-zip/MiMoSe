from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from missingbench.models.abstract_model import AbstractModel


basic_dims = 16
num_modals = 4
input_patch_size = 80
DATASET_MODALITY_ORDER = (2, 0, 1, 3)


def normalization(planes: int, norm: str = "bn") -> nn.Module:
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
    def __init__(self, in_channel: int = 64, num_cls: int = 4, num_modal: int = 4) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
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
    def __init__(self, in_channel: int = 64, num_cls: int = 4, num_modal: int = 4) -> None:
        super().__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_modal, in_channel // 4, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x2.size()
        masked = torch.zeros_like(x2)
        masked[mask, ...] = x2[mask, ...]
        masked = masked.view(batch_size, -1, height, width, depth)
        return self.prm_layer(torch.cat((x1, self.embedding_layer(masked)), dim=1))


class ModalFusion(nn.Module):
    def __init__(self, in_channel: int = 64, num_modal: int = 4) -> None:
        super().__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(num_modal * in_channel + 1, 128, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(128, num_modal, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, prm: torch.Tensor) -> torch.Tensor:
        batch_size, modalities, channels, _, _, _ = x.size()
        prm_avg = torch.mean(prm, dim=(3, 4, 5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg
        feat_avg = feat_avg.view(batch_size, modalities * channels, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(batch_size, 1, 1, 1, 1)), dim=1)
        weight = self.weight_layer(feat_avg).view(batch_size, modalities, 1)
        weight = self.sigmoid(weight).view(batch_size, modalities, 1, 1, 1, 1)
        return torch.sum(x * weight, dim=1)


class RegionFusion(nn.Module):
    def __init__(self, in_channel: int = 64, num_cls: int = 4) -> None:
        super().__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
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
        self.modal_fusion = nn.ModuleList([ModalFusion(in_channel=in_channel) for _ in range(num_cls)])
        self.region_fusion = RegionFusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
            GeneralConv3d(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1),
        )

    def forward(self, x: torch.Tensor, prm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, height, width, depth = x.size()
        masked = torch.zeros_like(x)
        masked[mask, ...] = x[mask, ...]

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, x.size(2), 1, 1, 1)
        flair = masked[:, 0:1, ...] * prm
        t1ce = masked[:, 1:2, ...] * prm
        t1 = masked[:, 2:3, ...] * prm
        t2 = masked[:, 3:4, ...] * prm
        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        region_fused_feat = []
        for index in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[index](region_feat[index], prm[:, index : index + 1, ...]))
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

        self.e2_c1 = GeneralConv3d(basic_dims, basic_dims * 2, stride=2, pad_type="reflect")
        self.e2_c2 = GeneralConv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = GeneralConv3d(basic_dims * 2, basic_dims * 2, pad_type="reflect")

        self.e3_c1 = GeneralConv3d(basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect")
        self.e3_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = GeneralConv3d(basic_dims * 4, basic_dims * 4, pad_type="reflect")

        self.e4_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect")
        self.e4_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = GeneralConv3d(basic_dims * 8, basic_dims * 8, pad_type="reflect")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.d3_out = GeneralConv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = GeneralConv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
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
    def __init__(self, num_cls: int = 4) -> None:
        super().__init__()
        self.d3_c1 = GeneralConv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = GeneralConv3d(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = GeneralConv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2_c1 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = GeneralConv3d(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = GeneralConv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1_c1 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = GeneralConv3d(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = GeneralConv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")

        self.seg_layer = nn.Conv3d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True)

        self.RFM4 = RegionAwareModalFusion(in_channel=basic_dims * 8, num_cls=num_cls)
        self.RFM3 = RegionAwareModalFusion(in_channel=basic_dims * 4, num_cls=num_cls)
        self.RFM2 = RegionAwareModalFusion(in_channel=basic_dims * 2, num_cls=num_cls)
        self.RFM1 = RegionAwareModalFusion(in_channel=basic_dims, num_cls=num_cls)

        self.prm_generator4 = PRMGeneratorLastStage(in_channel=basic_dims * 8, num_cls=num_cls)
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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        fusion_x4 = de_x4
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

        pred = self.softmax(self.seg_layer(de_x1))
        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4)), fusion_x4


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        relu: bool = True,
        norm: bool = True,
        bias: bool = False,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.drop = nn.Dropout3d(p=drop_rate) if drop_rate != 0 else None
        self.norm = nn.InstanceNorm3d(out_planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.drop is not None:
            x = self.drop(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class AdaptiveInstanceNorm(nn.Module):
    def forward(
        self,
        content: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        del epsilon
        content_mean = torch.mean(content, [2, 3, 4], keepdim=True)
        content_std = torch.std(content, [2, 3, 4], keepdim=True)
        return gamma * ((content - content_mean) / content_std) + beta


class AdaptiveResBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int) -> None:
        super().__init__()
        self.conv1 = BasicConv(in_planes, out_planes, 3, stride=1, padding=1, relu=False, norm=False)
        self.i_norm1 = AdaptiveInstanceNorm()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = BasicConv(out_planes, out_planes, 3, stride=1, padding=1, relu=False, norm=False)
        self.i_norm2 = AdaptiveInstanceNorm()

    def forward(self, x_init: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x_init)
        x = self.i_norm1(x, sigma, mu)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.i_norm2(x, sigma, mu)
        return x + x_init


class StyleEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, n_base_ch_se: int = 32) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            BasicConv(in_channels, n_base_ch_se, 7, stride=1, padding=3, relu=True, norm=False),
            BasicConv(n_base_ch_se, n_base_ch_se * 2, 4, stride=2, padding=1, relu=True, norm=False),
            BasicConv(n_base_ch_se * 2, n_base_ch_se * 4, 4, stride=2, padding=1, relu=True, norm=False),
            BasicConv(n_base_ch_se * 4, n_base_ch_se * 4, 4, stride=2, padding=1, relu=True, norm=False),
            BasicConv(n_base_ch_se * 4, n_base_ch_se * 4, 4, stride=2, padding=1, relu=True, norm=False),
        ]
        self.encoder = nn.Sequential(*layers)
        self.final_conv = BasicConv(n_base_ch_se * 4, n_base_ch_se * 4, 1, stride=2, padding=0, relu=False, norm=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.mean(x, [2, 3, 4], keepdim=True)
        return self.final_conv(x)


class MLP(nn.Module):
    def __init__(self, in_ch: int = 128, mlp_ch: int = 128) -> None:
        super().__init__()
        self.channel = mlp_ch
        self.l1 = nn.Linear(in_ch, mlp_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(mlp_ch, mlp_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.l_mu = nn.Linear(mlp_ch, mlp_ch)
        self.l_sigma = nn.Linear(mlp_ch, mlp_ch)

    def forward(self, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = style.view(style.size(0), -1)
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        mu = self.l_mu(x).reshape(-1, self.channel, 1, 1, 1)
        sigma = self.l_sigma(x).reshape(-1, self.channel, 1, 1, 1)
        return mu, sigma


class ImageDecoder(nn.Module):
    def __init__(
        self,
        in_style_ch: int = 128,
        in_content_ch: int = 128,
        mlp_ch: int = 128,
        img_ch: int = 1,
    ) -> None:
        super().__init__()
        channel = mlp_ch
        self.mlp = MLP(in_style_ch, mlp_ch)
        self.res_blocks = nn.ModuleList([AdaptiveResBlock(in_content_ch, channel) for _ in range(4)])

        decoder_blocks: list[nn.Module] = []
        for _ in range(3):
            decoder_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="trilinear"),
                    BasicConv(channel, channel // 2, 5, stride=1, padding=2, relu=False, norm=False),
                )
            )
            channel = channel // 2
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.final_conv = BasicConv(channel, img_ch, 7, stride=1, padding=3, relu=False, norm=False)

    def forward(self, style: torch.Tensor, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.mlp(style)
        x = content
        for block in self.res_blocks:
            x = block(x, mu, sigma)
        for block in self.decoder_blocks:
            x = block(x)
            x = F.layer_norm(x, x.shape[1:])
            x = F.relu(x, inplace=True)
        return self.final_conv(x), mu, sigma


class DCSeg(AbstractModel):
    def __init__(self, num_cls: int = 4, fusion_type: str = "RFM") -> None:
        super().__init__()
        self.fusion_type = fusion_type

        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.flair_style_encoder = StyleEncoder()
        self.t1ce_style_encoder = StyleEncoder()
        self.t1_style_encoder = StyleEncoder()
        self.t2_style_encoder = StyleEncoder()

        self.flair_decoder = ImageDecoder()
        self.t1ce_decoder = ImageDecoder()
        self.t1_decoder = ImageDecoder()
        self.t2_decoder = ImageDecoder()
        self.decoders = nn.ModuleList(
            [self.flair_decoder, self.t1ce_decoder, self.t1_decoder, self.t2_decoder]
        )

        self.decoder_fuse = DecoderFuse(num_cls=num_cls)
        self.decoder_sep = DecoderSep(num_cls=num_cls)

        in_out_ch = 128
        self.content_attn = BasicConv(in_out_ch * 4, 4, 3, stride=1, padding=1, relu=False, norm=True)
        self.content_share_conv_list = BasicConv(in_out_ch * 4, in_out_ch, 1, stride=1, padding=0, relu=True, norm=True)

        self.is_training = False

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        x, mask = self._remap_input_order(x, mask)

        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1)
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        fuse_pred, prm_preds, fusion_x4 = self.decoder_fuse(x1, x2, x3, x4, mask)
        if not self.is_training:
            return fuse_pred

        flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
        t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
        t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
        t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)

        flair_style = self.flair_style_encoder(x[:, 0:1, :, :, :])
        t1ce_style = self.t1ce_style_encoder(x[:, 1:2, :, :, :])
        t1_style = self.t1_style_encoder(x[:, 2:3, :, :, :])
        t2_style = self.t2_style_encoder(x[:, 3:4, :, :, :])

        if self.fusion_type == "RFM":
            out = fusion_x4
        elif self.fusion_type == "gated":
            share_concat = torch.cat([flair_x4, t1ce_x4, t1_x4, t2_x4], 1)
            attnmap = torch.sigmoid(self.content_attn(share_concat))
            share_content = []
            for index in range(num_modals):
                share_content.append(x4[:, index] * attnmap[:, index : index + 1])
            out = self.content_share_conv_list(torch.cat(share_content, 1))
        else:
            out = x4.mean(1)

        recon_list: list[torch.Tensor] = []
        mu_list: list[torch.Tensor] = []
        sigma_list: list[torch.Tensor] = []
        for index, style in enumerate([flair_style, t1ce_style, t1_style, t2_style]):
            recon, mu, sigma = self.decoders[index](style, out)
            recon_list.append(recon)
            mu_list.append(mu)
            sigma_list.append(sigma)
        recon_out = torch.cat(recon_list, 1)

        contents = x4
        styles = torch.stack([flair_style, t1ce_style, t1_style, t2_style], 1)
        styles = styles.squeeze(-1).squeeze(-1).squeeze(-1)
        return (
            fuse_pred,
            (flair_pred, t1ce_pred, t1_pred, t2_pred),
            prm_preds,
            recon_out,
            mu_list,
            sigma_list,
            contents,
            styles,
        )

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (input_patch_size, input_patch_size, input_patch_size):
            return self(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.decoder_fuse.seg_layer.out_channels,
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
                    patch = images[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size]
                    patch_pred = self(patch, mask)
                    prediction[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size] += patch_pred
                    weight[:, :, h : h + input_patch_size, w : w + input_patch_size, d : d + input_patch_size] += 1
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

        stride = int(input_patch_size * (1 - 0.5))
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
            raise RuntimeError(
                f"DCSeg expects {num_modals} input modalities, got {images.size(1)}"
            )
        if mask.ndim != 2 or mask.size(1) != num_modals:
            raise RuntimeError(
                f"DCSeg expects mask shape [B, {num_modals}], got {tuple(mask.shape)}"
            )

        # External pipeline uses [t1c, t1n, t2f, t2w]; DC-Seg internals expect
        # [flair, t1ce, t1, t2].
        return images[:, DATASET_MODALITY_ORDER, ...], mask[:, DATASET_MODALITY_ORDER]


Model = DCSeg


__all__ = ["DCSeg", "Model"]
