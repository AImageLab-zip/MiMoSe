import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from dcn.modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from einops import rearrange
import numbers


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='reflect', norm='in', act_type='lrelu', relufactor=0.2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(in_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x
    

def to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')

def to_4d(x,d,h,w):
    return rearrange(x, 'b (d h w) c -> b c d h w',d=d,h=h,w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        d, h, w = x.shape[-3:]
        return to_4d(self.body(to_3d(x)), d, h, w)


class IFN(nn.Module):
    def __init__(self, dim, bias):
        super(IFN, self).__init__()
        hidden_dims = 340
        self.project_in = nn.Conv3d(dim, hidden_dims*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(hidden_dims*2, hidden_dims*2, kernel_size=3, stride=1, padding=1, groups=hidden_dims*2, bias=bias)
        self.project_out = nn.Conv3d(hidden_dims, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Chan_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Chan_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        b,c,d,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out)
        return out
    

class Spatial_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Spatial_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
    

    def forward(self, x):
        b,c,d,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)  #B,C,N
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)  #B,C,N
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)  #B,C,N

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (k.transpose(-2, -1) @ q) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (v @ attn)
        
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out)
        return out


class GlobalBlock(nn.Module):
    def __init__(self, dim, num_heads,  bias):
        super(GlobalBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.chan_attn = Chan_Attention(dim, num_heads, bias)
        self.spatial_attn = Spatial_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ifn = IFN(dim, bias)

    def forward(self, x):
        norm_x = self.norm1(x)
        x1 = self.chan_attn(norm_x)
        x2 = self.spatial_attn(norm_x)
        x = x + x1 + x2
        x = x + self.ifn(self.norm2(x))
        return x


class dap(nn.Module):
    def __init__(self, in_channel, reduction=2, num_cls=2):
        super(dap, self).__init__()

        self.conv_first = ConvBlock(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1)
        self.non_attention = ConvBlock(in_channel, in_channel, k_size=3, padding=1, stride=1)
        self.conv_last =  ConvBlock(in_channel, in_channel, k_size=1, padding=0, stride=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.fc1 = nn.Linear(in_channel * num_cls, in_channel // reduction, bias=True)
        self.fc2 = nn.Linear(in_channel // reduction, 2, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.deformable_conv = DeformConvPack(in_channel, in_channel, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        B, C, H, W, D = x.shape

        avg_squeeze_tensor = self.global_avg_pool(x)
        avg_fc_out = self.fc1(avg_squeeze_tensor.view(B, C))
        shared_fc_out = self.lrelu(avg_fc_out)
        dynamic_weight = self.softmax(self.fc2(shared_fc_out))

        first_conv_out = self.conv_first(x)
        belta = self.deformable_conv(first_conv_out)
        alpha = self.non_attention(first_conv_out)
        last_conv_out = belta * dynamic_weight[:, 0].view(B, 1, 1, 1, 1) + alpha * dynamic_weight[:, 1].view(B, 1, 1, 1, 1)
        out = self.conv_last(last_conv_out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=1, basic_dims=8):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=in_channels, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                               bias=True)
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


class recblock(nn.Module):
    def __init__(self, num_cls=4, basic_dims=8):
        super(recblock, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = ConvBlock(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = ConvBlock(basic_dims * 16, basic_dims * 8)
        self.d4_out = ConvBlock(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = ConvBlock(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = ConvBlock(basic_dims * 8, basic_dims * 4)
        self.d3_out = ConvBlock(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = ConvBlock(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = ConvBlock(basic_dims * 4, basic_dims * 2)
        self.d2_out = ConvBlock(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = ConvBlock(basic_dims * 2, basic_dims)
        self.d1_c2 = ConvBlock(basic_dims * 2, basic_dims)
        self.d1_out = ConvBlock(basic_dims, basic_dims, k_size=1, padding=0)
        
        self.layer = nn.Conv3d(in_channels=basic_dims, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)


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

        logits = self.layer(de_x1)
        # pred = self.softmax(logits)

        return logits
    

class Decoder(nn.Module):
    def __init__(self, num_cls=4, basic_dims=8):
        super(Decoder, self).__init__()

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

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.block5 = dap(in_channel=basic_dims * 16, num_cls=num_cls)
        self.block4 = dap(in_channel=basic_dims * 8, num_cls=num_cls)
        self.block3 = dap(in_channel=basic_dims * 4, num_cls=num_cls)
        self.block2 = dap(in_channel=basic_dims * 2, num_cls=num_cls)
        self.block1 = dap(in_channel=basic_dims * 1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.block5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.block4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.block3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.block2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.block1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()
        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()

        self.flair_glo = GlobalBlock(dim=128, num_heads=8, bias=False)
        self.t1ce_glo = GlobalBlock(dim=128, num_heads=8, bias=False)
        self.t1_glo = GlobalBlock(dim=128, num_heads=8, bias=False)
        self.t2_glo = GlobalBlock(dim=128, num_heads=8,  bias=False)

        self.rec = recblock()

        self.masker = MaskModal()
        self.decoder = Decoder(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)  

    def forward(self, x, mask):
        flair = x[:, 0:1, :, :, :]
        t1ce = x[:, 1:2, :, :, :]
        t1 = x[:, 2:3, :, :, :]
        t2 = x[:, 3:4, :, :, :]
        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(flair)
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(t1ce)
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(t1)
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(t2)

        flair_lg5 = self.flair_glo(flair_x5)
        t1ce_lg5 = self.t1ce_glo(t1ce_x5)
        t1_lg5 = self.t1_glo(t1_x5)
        t2_lg5 = self.t2_glo(t2_x5)

        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)  # Bx4xCxHWZ
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5 = self.masker(torch.stack((flair_lg5, t1ce_lg5, t1_lg5, t2_lg5), dim=1), mask)

        pred, preds = self.decoder(x1, x2, x3, x4, x5)

        if self.is_training:    
            flair_rec = self.rec(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_rec = self.rec(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_rec = self.rec(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_rec = self.rec(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)

            recs = [flair_rec,t1ce_rec,t1_rec,t2_rec]
            return pred, preds, recs
        
        return pred


if __name__ == '__main__':
    # x.shape [1, 4, 128, 128, 128]
    # y.shape [1, 4, 128, 128, 128]
    # mask.shape [1, 4]
    x = torch.rand(1, 4, 128, 128, 128)
    model = Model()
    print(model)
    masks = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False],
                      [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True],
                      [False, True, True, False], [False, True, False, True], [False, False, True, True],
                      [True, True, True, False], [True, True, False, True], [True, False, True, True],
                      [False, True, True, True],
                      [True, True, True, True]])
    mask = torch.unsqueeze(torch.from_numpy(masks[2]), dim=0)
    a, b, c = model(x, mask)
    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_millions = total_params / 1_000_000
    print(f"Total Parameters: {total_params_in_millions} M")
    print(a.shape)
    for i in b:
        print(i.shape)

