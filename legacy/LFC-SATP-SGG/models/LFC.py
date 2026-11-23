import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (general_conv3d, normalization, prm_generator, prm_fusion,
                    prm_generator_laststage, region_aware_modal_fusion, fusion_postnorm, ada)
from models.blocks import nchwd2nlc2nchwd, DepthWiseConvBlock, ResBlock, GroupConvBlock, MultiMaskAttentionLayer, \
    MultiMaskCrossBlock
from torch.nn.init import constant_, xavier_uniform_
from models.mask import mask_gen_fusion, mask_gen_skip
import clip
import numpy as np
from typing import Tuple, Any

basic_dims = 16
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 3
num_modals = 4
patch_size = 5
HWD = 80
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DilatedConv3dBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dilation=2, norm='in', act_type='lrelu', relufactor=0.2):
        super().__init__()

        padding = dilation
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1,
                              padding=padding, dilation=dilation, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class NonLocalBlock3D(nn.Module):

    def __init__(self, in_ch, inter_ch=None):
        super().__init__()
        if inter_ch is None:
            inter_ch = max(in_ch // 2, 1)

        self.theta = nn.Conv3d(in_ch, inter_ch, kernel_size=1)
        self.phi = nn.Conv3d(in_ch, inter_ch, kernel_size=1)
        self.g = nn.Conv3d(in_ch, in_ch, kernel_size=1)
        self.Wz = nn.Conv3d(in_ch, in_ch, kernel_size=1)

        nn.init.zeros_(self.Wz.weight)
        nn.init.zeros_(self.Wz.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape

        theta = self.theta(x).view(B, -1, D * H * W)
        phi = self.phi(x).view(B, -1, D * H * W)
        g = self.g(x).view(B, -1, D * H * W)

        attn = torch.matmul(theta.transpose(1, 2), phi)
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(g, attn.transpose(1, 2))
        y = y.view(B, C, D, H, W)

        y = self.Wz(y)
        return x + y


class Encoder_ContextEnhanced(nn.Module):

    def __init__(self, basic_dims=16):
        super().__init__()
        ch1, ch2, ch3, ch4, ch5 = basic_dims, basic_dims * 2, basic_dims * 4, basic_dims * 8, basic_dims * 16

        self.e1_c1 = general_conv3d(1, ch1, pad_type='reflect')
        self.e1_c2 = general_conv3d(ch1, ch1, pad_type='reflect')
        self.e1_c3 = general_conv3d(ch1, ch1, pad_type='reflect')

        self.e2_c1 = general_conv3d(ch1, ch2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(ch2, ch2, pad_type='reflect')
        self.e2_c3 = general_conv3d(ch2, ch2, pad_type='reflect')

        self.e3_c1 = general_conv3d(ch2, ch3, stride=2, pad_type='reflect')
        self.e3_c2 = DilatedConv3dBlock(ch3, ch3, dilation=1)
        self.e3_c3 = DilatedConv3dBlock(ch3, ch3, dilation=2)

        self.e4_c1 = general_conv3d(ch3, ch4, stride=2, pad_type='reflect')
        self.e4_c2 = DilatedConv3dBlock(ch4, ch4, dilation=2)
        self.e4_c3 = DilatedConv3dBlock(ch4, ch4, dilation=3)

        self.e5_c1 = general_conv3d(ch4, ch5, stride=2, pad_type='reflect')
        self.e5_c2_nonlocal = NonLocalBlock3D(ch5)
        self.e5_c3 = general_conv3d(ch5, ch5, pad_type='reflect')

        self.gamma_nl = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x1 = self.e1_c1(x);
        x1 = x1 + self.e1_c3(self.e1_c2(x1))
        x2 = self.e2_c1(x1);
        x2 = x2 + self.e2_c3(self.e2_c2(x2))
        x3 = self.e3_c1(x2);
        x3 = x3 + self.e3_c3(self.e3_c2(x3))
        x4 = self.e4_c1(x3);
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)

        nl_out = self.e5_c2_nonlocal(x5)
        delta = nl_out - x5
        x5 = x5 + self.gamma_nl * self.e5_c3(delta)

        return x1, x2, x3, x4, x5


class AttentionGate3D(nn.Module):

    def __init__(self, F_g: int, F_l: int, F_int: int, max_groups: int = 32):
        super().__init__()

        def make_gn(num_channels: int, max_groups: int = 32) -> nn.GroupNorm:
            groups = min(max_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, num_channels)

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            make_gn(F_int, max_groups=max_groups)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            make_gn(F_int, max_groups=max_groups)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class CrossAttentionFusion(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm_feature = nn.LayerNorm(embed_dim)
        self.norm_condition = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, image_features, condition_vector):
        shortcut = image_features

        query = self.norm_feature(image_features)
        condition = self.norm_condition(condition_vector).unsqueeze(1)
        key = value = condition

        query = torch.nan_to_num(query)
        key = torch.nan_to_num(key)

        attn_output, attn_weights = self.attention(query=query, key=key, value=value)

        if torch.isnan(attn_output).any():
            print("NaN detected in CrossAttention output!")

            attn_output = torch.nan_to_num(attn_output)

        return shortcut + attn_output


def load_clip_model(model_name: str) -> Tuple[Any, Any]:
    CLIP_NAME = {
        "clip_vit_b32": "ViT-B/32",
        "clip_vit_b16": "ViT-B/16",
        "clip_resnet50": "RN50",
        "clip_resnet101": "RN101"
    }

    if model_name not in CLIP_NAME:
        raise ValueError(f"Unsupported model name: {model_name}. Available names: {list(CLIP_NAME.keys())}")

    clip_model, preprocess = clip.load(CLIP_NAME[model_name], device=device)
    tokenizer = clip.tokenize
    return clip_model, tokenizer


def generate_text_description(mask_tensor):
    modality_names = ["FLAIR", "T1ce", "T1", "T2"]
    descriptions = []
    for mask in mask_tensor:
        present_modalities = [name for i, name in enumerate(modality_names) if mask[i]]
        if not present_modalities:
            desc = "All modalities are missing."
        elif len(present_modalities) == 4:
            desc = "All modalities are present."
        else:
            desc = "Available modalities: " + ", ".join(present_modalities) + "."
        descriptions.append(desc)
    return descriptions


class RelativePositionEncoding(nn.Module):
    def __init__(self, num_buckets, max_distance, embedding_dim):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, embedding_dim)

    def forward(self, x_device, length_q, length_k):
        device = x_device.device

        context_position = torch.arange(length_q, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(length_k, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute(2, 0, 1).unsqueeze(0)
        return values

    def _relative_position_bucket(self, relative_position):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) /
                torch.log(torch.tensor(max_distance / max_exact, dtype=torch.float, device=n.device)) *
                (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.tensor(num_buckets - 1, dtype=torch.long, device=n.device))
        ret += torch.where(is_small, n, val_if_large)
        return ret


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e5_c1 = general_conv3d(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e5_c3 = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')

    def forward(self, x):
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


class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.d4_c1 = general_conv3d(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.d3_c1 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.d2_c1 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.d1_c1 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))

        de_x4 = self.d3_c1(self.d3(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))

        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))

        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred


class SoftHeMISFusion(nn.Module):
    def __init__(self, in_channel=64, eta_init=0.2):
        super().__init__()
        self.fuse = general_conv3d(in_channel * 2, in_channel, k_size=1, padding=0, stride=1)

        self.eta = nn.Parameter(torch.tensor(float(eta_init)).atanh())

    def forward(self, x, hard_mask, soft_w=None):

        B, K, C, D, H, W = x.shape
        dtype = x.dtype
        device = x.device

        m_hard = hard_mask.to(dtype=dtype, device=device).view(B, K, 1, 1, 1, 1)
        denom_hard = m_hard.sum(dim=1, keepdim=True).clamp_min(1.0)

        mu_present = (x * m_hard).sum(dim=1, keepdim=True) / denom_hard

        x_tilde = x * m_hard + (1.0 - m_hard) * mu_present

        eta = torch.sigmoid(self.eta)
        if soft_w is None:
            m_soft = (hard_mask.to(dtype) + (1.0 - hard_mask.to(dtype)) * eta).view(B, K, 1, 1, 1, 1)
        else:

            sw = soft_w.clamp(0, 1).to(dtype=dtype, device=device).view(B, K, 1, 1, 1, 1)
            m_soft = (hard_mask.to(dtype).view(B, K, 1, 1, 1, 1) + (
                        1.0 - hard_mask.to(dtype).view(B, K, 1, 1, 1, 1)) * sw)

        denom_soft = m_soft.sum(dim=1, keepdim=True).clamp_min(1.0)

        mu = (x_tilde * m_soft).sum(dim=1, keepdim=True) / denom_soft
        var = ((x_tilde - mu) ** 2 * m_soft).sum(dim=1, keepdim=True) / denom_soft

        feat = torch.cat([mu, var], dim=2).squeeze(1)
        return self.fuse(feat)


class msfs(nn.Module):
    def __init__(self, num_cls=4):
        super(msfs, self).__init__()

        self.d5_c2 = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.d5_out = general_conv3d(basic_dims * 16, basic_dims * 16, k_size=1, padding=0, pad_type='reflect')

        self.d4_c1 = general_conv3d(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv3d(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv3d(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=False)

        self.msm5 = SoftHeMISFusion(in_channel=basic_dims * 16)
        self.msm4 = SoftHeMISFusion(in_channel=basic_dims * 8)
        self.msm3 = SoftHeMISFusion(in_channel=basic_dims * 4)
        self.msm2 = SoftHeMISFusion(in_channel=basic_dims * 2)
        self.msm1 = SoftHeMISFusion(in_channel=basic_dims * 1)

        self.prm_fusion5 = prm_fusion(in_channel=basic_dims * 16, num_cls=num_cls)
        self.prm_fusion4 = prm_fusion(in_channel=basic_dims * 8, num_cls=num_cls)
        self.prm_fusion3 = prm_fusion(in_channel=basic_dims * 4, num_cls=num_cls)
        self.prm_fusion2 = prm_fusion(in_channel=basic_dims * 2, num_cls=num_cls)
        self.prm_fusion1 = prm_fusion(in_channel=basic_dims * 1, num_cls=num_cls)

        self.ag4 = AttentionGate3D(F_g=basic_dims * 8, F_l=basic_dims * 8, F_int=basic_dims * 4)
        self.ag3 = AttentionGate3D(F_g=basic_dims * 4, F_l=basic_dims * 4, F_int=basic_dims * 2)
        self.ag2 = AttentionGate3D(F_g=basic_dims * 2, F_l=basic_dims * 2, F_int=basic_dims * 1)
        self.ag1 = AttentionGate3D(F_g=basic_dims * 1, F_l=basic_dims * 1, F_int=basic_dims // 2)

    def forward(self, dx1, dx2, dx3, dx4, dx5, mask):
        de_x5_fused = self.msm5(dx5, mask)
        prm_pred5 = self.prm_fusion5(de_x5_fused)
        de_x5 = self.d5_out(self.d5_c2(de_x5_fused))

        de_x5_upsampled = self.d4_c1(self.up2(de_x5))
        de_x4_skip = self.msm4(dx4, mask)
        prm_pred4 = self.prm_fusion4(de_x5_upsampled)
        de_x4_att = self.ag4(g=de_x5_upsampled, x=de_x4_skip)

        de_x4_cat = torch.cat((de_x4_att, de_x5_upsampled), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4_cat))

        de_x4_upsampled = self.d3_c1(self.up2(de_x4))
        de_x3_skip = self.msm3(dx3, mask)
        prm_pred3 = self.prm_fusion3(de_x4_upsampled)

        de_x3_att = self.ag3(g=de_x4_upsampled, x=de_x3_skip)
        de_x3_cat = torch.cat((de_x3_att, de_x4_upsampled), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3_cat))

        de_x3_upsampled = self.d2_c1(self.up2(de_x3))
        de_x2_skip = self.msm2(dx2, mask)
        prm_pred2 = self.prm_fusion2(de_x3_upsampled)
        de_x2_att = self.ag2(g=de_x3_upsampled, x=de_x2_skip)
        de_x2_cat = torch.cat((de_x2_att, de_x3_upsampled), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2_cat))

        de_x2_upsampled = self.d1_c1(self.up2(de_x2))
        de_x1_skip = self.msm1(dx1, mask)
        prm_pred1 = self.prm_fusion1(de_x2_upsampled)
        de_x1_att = self.ag1(g=de_x2_upsampled, x=de_x1_skip)
        de_x1_cat = torch.cat((de_x1_att, de_x2_upsampled), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1_cat))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4), self.up16(prm_pred5))


@torch.no_grad()
def mask_gen_for_pgn(Batchsize, NumHead, patches, NumClass, mask):
    device = mask.device
    B, C = Batchsize, NumClass
    L = patches * C

    present_modal = (mask > 0).to(torch.bool)
    present_tokens = present_modal.unsqueeze(-1).expand(B, C, patches).reshape(B, L)

    row_is_missing = (~present_tokens).unsqueeze(2)
    col_is_present = present_tokens.unsqueeze(1)

    attn_mask = col_is_present.repeat(1, L, 1)

    eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    attn_mask |= eye

    return attn_mask.unsqueeze(1).expand(-1, NumHead, -1, -1).contiguous()


@torch.no_grad()
def mask_gen_fusion_with_text(Batchsize, NumHead, patches, NumClass, mask):
    device = mask.device
    B, C = Batchsize, NumClass
    L_no_text = patches * C
    L = 1 + L_no_text

    present_modal = (mask > 0).to(torch.bool)

    present_tokens_no_text = present_modal.unsqueeze(-1).expand(B, C, patches).reshape(B, L_no_text)

    row_present = torch.zeros(B, L, dtype=torch.bool, device=device)
    col_present = torch.zeros(B, L, dtype=torch.bool, device=device)
    row_present[:, 1:] = present_tokens_no_text
    col_present[:, 1:] = present_tokens_no_text

    attn_mask = torch.zeros(B, L, L, dtype=torch.bool, device=device)

    eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)  # [1, L, L]
    attn_mask |= eye
    attn_mask[:, :, 0] = True

    attn_mask |= (row_present.unsqueeze(2) & col_present.unsqueeze(1))

    row_missing = (~row_present)
    attn_mask |= (row_missing.unsqueeze(2) & col_present.unsqueeze(1))

    attn_mask = attn_mask.unsqueeze(1).expand(B, NumHead, L, L).contiguous()
    return attn_mask


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        x = self.net(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5
        return x


class MaskedResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, mask):
        y, attn = self.fn(x, mask)
        return y + x, attn


class MaskedPreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x, mask):
        x = self.norm(x)
        x, attn = self.fn(x, mask)
        return self.dropout(x), attn


class MaskedAttentionForLFC(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, dropout_rate=0.0,
            num_class=4, num_buckets=32, max_distance=128
    ):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.num_class = num_class

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        self.relative_position_encoding = RelativePositionEncoding(
            num_buckets, max_distance, heads
        )

        self.pre_ln = nn.LayerNorm(dim)

    def forward(self, x, mask):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        x = self.pre_ln(x)
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = (D ** -0.5)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # [B,H,N,N]
        rel_bias = self.relative_position_encoding(x, N, N).to(logits.device).to(logits.dtype)  # [1,H,N,N]
        logits = logits + rel_bias

        attn_mask = mask_gen_for_pgn(B, H, N // self.num_class, self.num_class, mask).to(device, non_blocking=True)
        logits = logits.masked_fill(~attn_mask, -1e4)

        attn = torch.softmax(logits, dim=-1).to(dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class MaskedAttentionForSATP(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, dropout_rate=0.0,
            num_class=4, num_buckets=32, max_distance=128
    ):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.num_class = num_class

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.relative_position_encoding = RelativePositionEncoding(
            num_buckets, max_distance, heads
        )
        self.pre_ln = nn.LayerNorm(dim)

    def forward(self, x, mask):
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        x = self.pre_ln(x)

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (D ** -0.5)
        logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

        rel_bias = self.relative_position_encoding(x, N, N).to(logits.device).to(logits.dtype)
        logits = logits + rel_bias

        attn_mask = mask_gen_fusion_with_text(
            B, H, (N - 1) // self.num_class, self.num_class, mask
        ).to(device, non_blocking=True)
        logits = logits.masked_fill(~attn_mask, -1e4)

        attn = torch.softmax(logits, dim=-1).to(dtype)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


class MaskedInterationForLFC(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                MaskedResidual(
                    MaskedPreNormDrop(
                        embedding_dim,
                        dropout_rate,

                        MaskedAttentionForLFC(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, mask):
        attn_list = []
        for j in range(self.depth):
            x, attn = self.cross_attention_list[j](x, mask)
            attn_list.append(attn.detach())
            x = self.cross_ffn_list[j](x)
        return x, attn_list


class TextGuidedMaskedInteration(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1):
        super().__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for j in range(self.depth):
            self.cross_attention_list.append(
                MaskedResidual(
                    MaskedPreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        MaskedAttentionForSATP(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )
        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, mask):
        attn_list = []
        for j in range(self.depth):
            x, attn = self.cross_attention_list[j](x, mask)
            attn_list.append(attn.detach())
            x = self.cross_ffn_list[j](x)
        return x, attn_list


class SATP(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_bottle = TextGuidedMaskedInteration(
            embedding_dim=basic_dims * 16, depth=depth, heads=num_heads, mlp_dim=mlp_dim
        )
        self.num_cls = num_modals
        self.text_pos_embed = nn.Parameter(torch.randn(1, 1, basic_dims * 16) * 0.01)
        self.txt_gate = nn.Parameter(torch.tensor(0.2))

        self.image_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, mask, pos, text_embed):
        flair, t1ce, t1, t2 = x

        embed_flair = flair.flatten(2).transpose(1, 2).contiguous()
        embed_t1ce = t1ce.flatten(2).transpose(1, 2).contiguous()
        embed_t1 = t1.flatten(2).transpose(1, 2).contiguous()
        embed_t2 = t2.flatten(2).transpose(1, 2).contiguous()

        embed_flair = embed_flair * self.image_scale
        embed_t1ce = embed_t1ce * self.image_scale
        embed_t1 = embed_t1 * self.image_scale
        embed_t2 = embed_t2 * self.image_scale

        text_token = text_embed.unsqueeze(1)
        image_tokens = torch.cat((embed_flair, embed_t1ce, embed_t1, embed_t2), dim=1)
        image_tokens = F.layer_norm(image_tokens, (image_tokens.size(-1),))
        embed_with_text = torch.cat((self.txt_gate * text_token, image_tokens), dim=1)

        pos_with_text = torch.cat((self.text_pos_embed, pos), dim=1)
        pos_with_text = F.normalize(pos_with_text, dim=-1)
        embed_with_text = embed_with_text + pos_with_text

        trans_output, attn = self.trans_bottle(embed_with_text, mask)
        image_tokens_out = trans_output[:, 1:, :]

        flair_trans, t1ce_trans, t1_trans, t2_trans = torch.chunk(
            image_tokens_out, self.num_cls, dim=1
        )
        return flair_trans, t1ce_trans, t1_trans, t2_trans, attn


class LFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cls = num_modals
        self.trans_bottle = MaskedInterationForLFC(
            embedding_dim=basic_dims * 16, depth=depth, heads=num_heads, mlp_dim=mlp_dim
        )

        self.query_generator = nn.Sequential(
            nn.Linear(basic_dims * 16, basic_dims * 16),
            nn.LayerNorm(basic_dims * 16),
            nn.GELU(),
            nn.Linear(basic_dims * 16, basic_dims * 16)
        )

        self.missing_aware_proj = nn.Sequential(
            nn.Linear(num_modals, basic_dims * 16),
            nn.LayerNorm(basic_dims * 16),
            nn.GELU()
        )
        self.fusion_modules = nn.ModuleList([
            CrossAttentionFusion(embed_dim=basic_dims * 16, num_heads=num_heads)
            for _ in range(num_modals)
        ])

    def forward(self, x, mask, pos):
        batch_size = x[0].size(0)
        seq_len = patch_size ** 3
        channels = basic_dims * 16
        modal_features = [m.flatten(2).transpose(1, 2).contiguous() for m in x]

        present_mask = mask.float().unsqueeze(-1).unsqueeze(-1)
        modal_means = []
        for i in range(self.num_cls):
            feat_mean = modal_features[i].mean(dim=1, keepdim=True)
            modal_means.append(feat_mean * present_mask[:, i])
        global_mean = torch.stack(modal_means, dim=1).sum(dim=1)

        query_tokens = self.query_generator(global_mean)
        query_tokens = query_tokens.repeat(1, seq_len, 1)

        modal_embeds = []
        for i in range(self.num_cls):
            present_features = modal_features[i]
            is_present = mask[:, i].view(-1, 1, 1).expand(-1, seq_len, channels)

            final_embed = torch.where(is_present, present_features, query_tokens)
            modal_embeds.append(final_embed)

        embed_cat = torch.cat(modal_embeds, dim=1)
        embed_cat = embed_cat + pos

        embed_cat = F.layer_norm(embed_cat, (embed_cat.size(-1),))

        generated_embeds, attn = self.trans_bottle(embed_cat, mask)
        generated_chunks = list(torch.chunk(generated_embeds, self.num_cls, dim=1))
        missing_mask = 1.0 - mask.float()
        global_missing_prompt = self.missing_aware_proj(missing_mask)
        final_outputs = []
        for i in range(self.num_cls):
            conditioned_feature = self.fusion_modules[i](generated_chunks[i], global_missing_prompt)

            final_outputs.append(conditioned_feature)

        return final_outputs[0], final_outputs[1], final_outputs[2], final_outputs[3], attn, missing_mask


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.ReLU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(B, C, D, H, W)
        return x


class SGG(nn.Module):
    def __init__(self, basic_dims=16, patch_size=5):
        super(SGG, self).__init__()
        self.basic_dims = basic_dims
        self.patch_size = patch_size

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.trans_flair = SimpleTransformerBlock(dim=basic_dims * 16)
        self.trans_t1ce = SimpleTransformerBlock(dim=basic_dims * 16)
        self.trans_t1 = SimpleTransformerBlock(dim=basic_dims * 16)
        self.trans_t2 = SimpleTransformerBlock(dim=basic_dims * 16)

    def _spatial_softmax(self, a: torch.Tensor, tau: float = 1.0, mix: float = 0.0) -> torch.Tensor:
        B, _, D, H, W = a.shape
        a = (a / tau).view(B, 1, -1)
        a = torch.softmax(a, dim=-1).view(B, 1, D, H, W)
        if mix > 0:
            a = (1 - mix) * a + mix * a.mean(dim=(2, 3, 4), keepdim=True)
        return a

    def forward(self, de_x1, de_x2, de_x3, de_x4, de_x5, attn=None):
        flair_tra, t1ce_tra, t1_tra, t2_tra = de_x5
        flair_x4, t1ce_x4, t1_x4, t2_x4 = de_x4
        flair_x3, t1ce_x3, t1_x3, t2_x3 = de_x3
        flair_x2, t1ce_x2, t1_x2, t2_x2 = de_x2
        flair_x1, t1ce_x1, t1_x1, t2_x1 = de_x1

        attn_flair = self.trans_flair(flair_tra)
        attn_t1ce = self.trans_t1ce(t1ce_tra)
        attn_t1 = self.trans_t1(t1_tra)
        attn_t2 = self.trans_t2(t2_tra)

        attn_flair = self._spatial_softmax(torch.sum(attn_flair, dim=1, keepdim=True), tau=0.7, mix=0.3)
        attn_t1ce = self._spatial_softmax(torch.sum(attn_t1ce, dim=1, keepdim=True), tau=0.7, mix=0.3)
        attn_t1 = self._spatial_softmax(torch.sum(attn_t1, dim=1, keepdim=True), tau=0.7, mix=0.3)
        attn_t2 = self._spatial_softmax(torch.sum(attn_t2, dim=1, keepdim=True), tau=0.7, mix=0.3)

        dex5 = (flair_tra * attn_flair, t1ce_tra * attn_t1ce, t1_tra * attn_t1, t2_tra * attn_t2)
        attn_flair = self.upsample(attn_flair)
        attn_t1ce = self.upsample(attn_t1ce)
        attn_t1 = self.upsample(attn_t1)
        attn_t2 = self.upsample(attn_t2)

        dex4 = (flair_x4 * attn_flair, t1ce_x4 * attn_t1ce, t1_x4 * attn_t1, t2_x4 * attn_t2)
        attn_flair = self.upsample(attn_flair)
        attn_t1ce = self.upsample(attn_t1ce)
        attn_t1 = self.upsample(attn_t1)
        attn_t2 = self.upsample(attn_t2)

        dex3 = (flair_x3 * attn_flair, t1ce_x3 * attn_t1ce, t1_x3 * attn_t1, t2_x3 * attn_t2)
        attn_flair = self.upsample(attn_flair)
        attn_t1ce = self.upsample(attn_t1ce)
        attn_t1 = self.upsample(attn_t1)
        attn_t2 = self.upsample(attn_t2)

        dex2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)
        dex1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        return dex1, dex2, dex3, dex4, dex5


class CrossAttention(nn.Module):
    def __init__(self, in_channels=basic_dims * 16, basic_dims=basic_dims):
        super(CrossAttention, self).__init__()

        self.query_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                    padding_mode='reflect')
        self.key_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                  padding_mode='reflect')
        self.value_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                    padding_mode='reflect')

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(p=0.1)
        self.out_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                  padding_mode='reflect')

    def forward(self, x, t):
        batch_size, C, D, H, W = x.size()

        query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)
        key = self.key_conv(t).view(batch_size, -1, D * H * W)
        value = self.value_conv(t).view(batch_size, -1, D * H * W).permute(0, 2, 1)

        scale = 1.0 / math.sqrt(query.size(-1))
        logits = torch.bmm(query, key) * scale
        attn = self.softmax(logits)
        attn = self.attn_drop(attn)

        out = torch.bmm(attn, value).permute(0, 2, 1).view(batch_size, C, D, H, W)
        out = self.out_conv(out)
        return out + x


class fusion(nn.Module):
    def __init__(self, in_channels=basic_dims * 32, basic_dims=basic_dims):
        super(fusion, self).__init__()

        self.cross_attention = CrossAttention(in_channels=basic_dims * 16, basic_dims=8)
        self.conv = nn.Conv3d(in_channels=basic_dims * 32, out_channels=basic_dims * 16, kernel_size=1, stride=1,
                              padding_mode='reflect')
        self.relu = nn.ReLU()

    def forward(self, x, t):
        x_cross = self.cross_attention(x, t)
        t_cross = self.cross_attention(t, x)

        combined = torch.cat((x_cross, t_cross), dim=1)

        out = self.conv(combined)
        out = self.relu(out)

        return out


def mask_to_index(mask):
    powers = torch.tensor([8, 4, 2, 1], device=mask.device)
    index = (mask.int() * powers).sum(dim=1)

    return index - 1


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.share_encoder = Encoder_ContextEnhanced()

        self.share1_encoder = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.share2_encoder = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.share3_encoder = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')

        self.share4_encoder = general_conv3d(basic_dims * 16, basic_dims * 16, pad_type='reflect')

        self.SATP = SATP()
        self.LFC = LFC()

        self.msfs = msfs(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.SGG = SGG()
        self.f1 = fusion()
        self.f2 = fusion()
        self.f3 = fusion()
        self.f4 = fusion()

        self.pos = nn.Parameter(torch.randn(1, (patch_size ** 3) * 4, basic_dims * 16) * 0.02)

        clip_model, tokenizer = load_clip_model("clip_vit_b32")
        for param in clip_model.parameters():
            param.requires_grad = False
        self.clip_model = clip_model.float().eval()
        self.tokenizer = tokenizer
        self.text_proj = nn.Linear(512, basic_dims * 16)

        self.fusion = nn.Parameter(nn.init.normal_(torch.zeros(1, patch_size ** 3, basic_dims * 16), mean=0.0, std=1.0))
        clip_model, tokenizer = load_clip_model("clip_vit_b32")
        for param in clip_model.parameters():
            param.requires_grad = False
        self.clip_model = clip_model.float().eval().to(device)
        self.tokenizer = tokenizer
        self.text_proj = nn.Linear(512, basic_dims * 16)
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):

            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask, text_desc_list=None):

        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])
        _, _, _, _, flair_s5 = self.share_encoder(x[:, 0:1, :, :, :])
        _, _, _, _, t1ce_s5 = self.share_encoder(x[:, 1:2, :, :, :])
        _, _, _, _, t1_s5 = self.share_encoder(x[:, 2:3, :, :, :])
        _, _, _, _, t2_s5 = self.share_encoder(x[:, 3:4, :, :, :])

        m_share = (flair_s5, t1ce_s5, t1_s5, t2_s5)

        B = x.size(0)

        text_desc_list = generate_text_description(mask)
        tokens = self.tokenizer(text_desc_list).to(x.device)
        with torch.no_grad():
            text_embed = self.clip_model.encode_text(tokens).float()
            text_embed = F.normalize(text_embed, dim=-1)
        text_embed = self.text_proj(text_embed)

        text_desc_list = generate_text_description(mask)
        tokens = self.tokenizer(text_desc_list).to(device)
        with torch.no_grad():
            text_embed = self.clip_model.encode_text(tokens).float()
            text_embed = F.normalize(text_embed, dim=-1)
        text_embed = self.text_proj(text_embed)

        flair_p, t1ce_p, t1_p, t2_p, attn, missing_mask = self.LFC(m_share, mask, self.pos)
        flair_prompt = flair_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                            3).contiguous()
        t1ce_prompt = t1ce_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                          3).contiguous()
        t1_prompt = t1_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                      3).contiguous()
        t2_prompt = t2_p.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                      3).contiguous()

        flair_p = self.share1_encoder(flair_prompt)
        t1ce_p = self.share2_encoder(t1ce_prompt)
        t1_p = self.share3_encoder(t1_prompt)
        t2_p = self.share4_encoder(t2_prompt)

        flair_x5 = self.f1(flair_x5, flair_p)
        t1ce_x5 = self.f2(t1ce_x5, t1ce_p)
        t1_x5 = self.f3(t1_x5, t1_p)
        t2_x5 = self.f4(t2_x5, t2_p)

        m_sbottle = (flair_x5, t1ce_x5, t1_x5, t2_x5)

        flair_trans, t1ce_trans, t1_trans, t2_trans, _ = self.SATP(m_sbottle, mask, self.pos, text_embed)

        flair_tra = flair_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                             3).contiguous()
        t1ce_tra = t1ce_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                           3).contiguous()
        t1_tra = t1_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                       3).contiguous()
        t2_tra = t2_trans.view(x.size(0), patch_size, patch_size, patch_size, basic_dims * 16).permute(0, 4, 1, 2,
                                                                                                       3).contiguous()

        de_x5 = (flair_tra, t1ce_tra, t1_tra, t2_tra)
        de_x4 = (flair_x4, t1ce_x4, t1_x4, t2_x4)
        de_x3 = (flair_x3, t1ce_x3, t1_x3, t2_x3)
        de_x2 = (flair_x2, t1ce_x2, t1_x2, t2_x2)
        de_x1 = (flair_x1, t1ce_x1, t1_x1, t2_x1)

        de_x1, de_x2, de_x3, de_x4, de_x5 = self.SGG(de_x1, de_x2, de_x3, de_x4, de_x5)

        de_x3 = torch.stack(de_x3, dim=1)
        de_x2 = torch.stack(de_x2, dim=1)
        de_x1 = torch.stack(de_x1, dim=1)
        de_x4 = torch.stack(de_x4, dim=1)
        de_x5 = torch.stack(de_x5, dim=1)

        fuse_pred, prm_preds = self.msfs(de_x1, de_x2, de_x3, de_x4, de_x5, mask)

        if self.training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds, (
                flair_prompt, t1ce_prompt, t1_prompt, t2_prompt)
        return fuse_pred