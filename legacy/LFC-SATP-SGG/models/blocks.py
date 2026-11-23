import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models.mask import mask_gen_cross4

basic_dims = 16
num_modals = 4


def nchwd2nlc2nchwd(module, x):
    B, C, H, W, D = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = module(x)
    x = x.transpose(1, 2).reshape(B, C, H, W, D).contiguous()
    return x

class DepthWiseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseConvBlock, self).__init__()
        mid_channels = in_channels
        self.conv1 = nn.Conv3d(in_channels,
                               mid_channels,
                               1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(mid_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv3d(mid_channels,
                               mid_channels,
                               3, 1, 1, groups=mid_channels)
        self.norm2 = layer_norm(mid_channels)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(mid_channels,
                               out_channels,
                               1, 1)
        self.norm3 = layer_norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.conv2(x)
        x = nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.conv3(x)
        x = nchwd2nlc2nchwd(self.norm3, x)
        return x

class GroupConvBlock(nn.Module):
    def __init__(self,
                 embed_dims=basic_dims,
                 expand_ratio=4,
                 proj_drop=0.):
        super(GroupConvBlock, self).__init__()
        self.pwconv1 = nn.Conv3d(embed_dims,
                                 embed_dims * expand_ratio,
                                 1, 1)
        layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = layer_norm(embed_dims * expand_ratio)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv3d(embed_dims * expand_ratio,
                                embed_dims * expand_ratio,
                                3, 1, 1, groups=embed_dims)
        self.norm2 = layer_norm(embed_dims * expand_ratio)
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Conv3d(embed_dims * expand_ratio,
                                 embed_dims,
                                 1, 1)
        self.norm3 = layer_norm(embed_dims)
        self.final_act = nn.GELU()
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, identity=None):
        input = x
        x = self.pwconv1(x)
        x = nchwd2nlc2nchwd(self.norm1, x)
        x = self.act1(x)

        x = self.dwconv(x)
        x = nchwd2nlc2nchwd(self.norm2, x)
        x = self.act2(x)

        x = self.pwconv2(x)
        x = nchwd2nlc2nchwd(self.norm3, x)

        if identity is None:
            x = input + self.proj_drop(x)
        else:
            x = identity + self.proj_drop(x)

        x = self.final_act(x)

        return x

class AttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=basic_dims,
                 query_dim=num_modals,
                 attn_drop=0.,
                 proj_drop=0.):
        super(AttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value):
        """x: B, C, H, W, D"""
        identity = query
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        key = self.key_map(key).flatten(2)
        value = self.value_map(value).flatten(2)

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)


class CrossBlock(nn.Module):
    def __init__(self,
                 feature_channels=basic_dims,
                 num_classes=num_modals,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(CrossBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = AttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps):
        kernels = self.cross_attn(query=kernels,
                                  key=feature_maps,
                                  value=feature_maps)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            feature_maps = self.ffn2(feature_maps, identity=feature_maps)

        return kernels, feature_maps

class ResBlock(nn.Module):
    def __init__(self, in_channels=4, channels=4):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv3d(in_channels, channels, 3, 1, 1)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1)
        if channels != in_channels:
            self.identity_map = nn.Conv3d(in_channels, channels, 1, 1, 0)
        else:
            self.identity_map = nn.Identity()

    def forward(self, x):
        # refer to paper
        # Identity Mapping in Deep Residual Networks
        out = nchwd2nlc2nchwd(self.norm1, x)
        out = self.act1(out)
        out = self.conv1(out)
        out = nchwd2nlc2nchwd(self.norm2, out)
        out = self.act2(out)
        out = self.conv2(out)
        out = out + self.identity_map(x)

        return out

class MultiMaskCrossBlock(nn.Module):
    def __init__(self,
                 feature_channels=basic_dims*16,
                 num_classes=basic_dims*16,
                 expand_ratio=4,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ffn_feature_maps=True):
        super(MultiMaskCrossBlock, self).__init__()
        self.ffn_feature_maps = ffn_feature_maps

        self.cross_attn = MultiMaskAttentionLayer(kv_dim=feature_channels,
                                         query_dim=num_classes,
                                         attn_drop=attn_drop_rate,
                                         proj_drop=drop_rate)

        if ffn_feature_maps:
            self.ffn2 = GroupConvBlock(embed_dims=feature_channels,
                                       expand_ratio=expand_ratio)
        self.ffn1 = GroupConvBlock(embed_dims=num_classes,
                                   expand_ratio=expand_ratio)

    def forward(self, kernels, feature_maps, mask):
        flair, t1ce, t1, t2 = feature_maps
        kernels = self.cross_attn(query = kernels,
                                  key = feature_maps,
                                  value = feature_maps,
                                  mask = mask)

        kernels = self.ffn1(kernels, identity=kernels)

        if self.ffn_feature_maps:
            flair = self.ffn2(flair, identity=flair)
            t1ce = self.ffn2(t1ce, identity=t1ce)
            t1 = self.ffn2(t1, identity=t1)
            t2 = self.ffn2(t2, identity=t2)
            feature_maps = (flair, t1ce, t1, t2)

        return kernels, feature_maps

class MultiMaskAttentionLayer(nn.Module):
    def __init__(self,
                 kv_dim=basic_dims,
                 query_dim=num_modals,
                 attn_drop=0.,
                 proj_drop=0.):
        super(MultiMaskAttentionLayer, self).__init__()
        self.attn_drop = nn.Dropout(attn_drop)
        self.query_map = DepthWiseConvBlock(query_dim, query_dim)
        self.key_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_flair = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1ce = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t1 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.key_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.value_map_t2 = DepthWiseConvBlock(kv_dim, kv_dim)
        self.out_project = DepthWiseConvBlock(query_dim, query_dim)

        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, query, key, value, mask):
        """x: B, C, H, W, D"""
        identity = query
        flair, t1ce, t1, t2 = key
        qb, qc, qh, qw, qd = query.shape
        query = self.query_map(query).flatten(2)
        key_flair = self.key_map_flair(flair).flatten(2)
        value_flair = self.value_map_flair(flair).flatten(2)
        key_t1ce = self.key_map_t1ce(t1ce).flatten(2)
        value_t1ce = self.value_map_t1ce(t1ce).flatten(2)
        key_t1 = self.key_map_t1(t1).flatten(2)
        value_t1 = self.value_map_t1(t1).flatten(2)
        key_t2 = self.key_map_t2(t2).flatten(2)
        value_t2 = self.value_map_t2(t2).flatten(2)

        key = torch.cat((key_flair, key_t1ce, key_t1, key_t2), dim=1)
        value = torch.cat((value_flair, value_t1ce, value_t1, value_t2), dim=1)

        kb, kc, kl = key.shape

        attn = (query @ key.transpose(-2, -1)) * (query.shape[-1]) ** -0.5
        self_mask = mask_gen_cross4(qb, qc, kc, mask).cuda(non_blocking=True)
        attn = attn.masked_fill(self_mask==0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ value
        x = x.reshape(qb, qc, qh, qw, qd)
        x = self.out_project(x)
        return identity + self.proj_drop(x)