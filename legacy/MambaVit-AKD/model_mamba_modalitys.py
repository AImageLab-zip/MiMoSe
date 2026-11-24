import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import Tuple, Union
from layers import normalization
from layers import general_conv3d
from layers import prm_generator_laststage, prm_generator, region_aware_modal_fusion
from monai.networks.nets import ViT
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from segMamba import SegMamba,initialize_model

basic_dims = 16
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

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Encoder0(nn.Module):
    def __init__(
            self,
            in_channels: int,
            img_size: Tuple[int, int, int],
            feature_size: int = 16,
            hidden_size: int = 384,
            mlp_dim: int = 2048,
            num_heads: int = 8,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            conv_block: bool = False,
            res_block: bool = True,
            dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False

        # Vision Transformer module
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=pos_embed,
            pos_embed_type="learnable",
            classification=self.classification,
            dropout_rate=dropout_rate,
        )

        # Encoder blocks
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=1,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )


    # 进行形状调整和维度的重排列
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))

        return enc1, enc2, enc3, enc4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4, modality_count=5):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

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

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, mask):
        prm_pred4 = self.prm_generator4(x4, mask)
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask)
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        #网络的最终预测结果+四个来自不同层级的 PRM预测结果
        return de_x1,logits,pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4))


class Model(nn.Module):
    def __init__(
            self,
            num_cls:int,
            in_channels: int,
            img_size: Tuple[int, int, int],
            hidden_size: int = 384,
            mlp_dim: int = 2048,
            num_heads: int = 8,
            pos_embed: str = "perceptron",
            dropout_rate: float = 0.0,
    ) -> None:
        super(Model, self).__init__()
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.flair_encoder0 = Encoder0(in_channels=in_channels, img_size=img_size, hidden_size=hidden_size,
                                     mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed,
                                     dropout_rate=dropout_rate, )
        self.t1ce_encoder0 = Encoder0(in_channels=in_channels, img_size=img_size, hidden_size=hidden_size,
                                    mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed,
                                    dropout_rate=dropout_rate, )
        self.t1_encoder0 = Encoder0(in_channels=in_channels, img_size=img_size, hidden_size=hidden_size,
                                  mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed,
                                  dropout_rate=dropout_rate, )
        self.t2_encoder0 = Encoder0(in_channels=in_channels, img_size=img_size, hidden_size=hidden_size,
                                  mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed,
                                  dropout_rate=dropout_rate, )
        self.m_encoder0 = Encoder0(in_channels=in_channels, img_size=img_size, hidden_size=hidden_size,
                                  mlp_dim=mlp_dim, num_heads=num_heads, pos_embed=pos_embed,
                                  dropout_rate=dropout_rate, )
        self.flair_encoder = Encoder(  )
        self.t1ce_encoder = Encoder(  )
        self.t1_encoder = Encoder(  )
        self.t2_encoder = Encoder(  )

        # # self.m_encoder = initialize_model()
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mask):
        ####把编码换成transfomer
        # print(x.shape)
        flair_x1, flair_x2, flair_x3, flair_x4 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4= self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self.t2_encoder(x[:, 3:4, :, :, :])
        

        # 根据 mask 选择模态并堆叠
        combined_modalities = None
        for i in range(mask[0].size(0)):  # mask 是一个布尔张量，长度为4
            if mask[0][i]:  # 如果当前模态被选中
                if combined_modalities is None:
                    combined_modalities = x[:, i:i + 1, :, :, :]  # 选择第 i 个模态
                else:
                    combined_modalities = torch.cat((combined_modalities, x[:, i:i + 1, :, :, :]), dim=1)  # 按照通道维度拼接

        # 检查是否有模态被选中
        if combined_modalities is None:
            raise ValueError("No modalities selected based on the mask.")

        # print(combined_modalities.shape) #torch.Size([1, 2, 96, 96, 96])
        combined_modalities=combined_modalities.cuda()
        mamba_encoder=initialize_model(combined_modalities)

        m_x1, m_x2, m_x3, m_x4 = mamba_encoder(combined_modalities)
        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1)
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        feature,logits,fuse_pred, prm_preds = self.decoder_fuse(x1, x2, x3, x4, mask)

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4)
            return feature,logits,fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds
        return feature,logits,fuse_pred



