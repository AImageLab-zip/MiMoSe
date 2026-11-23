import torch.nn as nn
from .encoder import Encoder3D
from .hgd_decoder import HGDDecoder

class SegModel(nn.Module):
    def __init__(self, in_ch=4, embed_dim=64, depth=(1,1,2,1), num_heads=(2,4,4,8), lora_r=8, num_experts=4, top_k=2, out_ch=4):
        super().__init__()
        self.enc = Encoder3D(in_ch=in_ch, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                             r=lora_r, num_experts=num_experts, top_k=top_k)
        c8  = in_ch*embed_dim
        c16 = in_ch*embed_dim*2
        c32 = in_ch*embed_dim*4
        self.dec = HGDDecoder(in_channels=(c8, c16, c32), mid=embed_dim*4, w=32, out_ch=out_ch)
    def forward(self, x):
        s8, s16, s32 = self.enc(x)
        logits = self.dec(s8, s16, s32)
        return logits
