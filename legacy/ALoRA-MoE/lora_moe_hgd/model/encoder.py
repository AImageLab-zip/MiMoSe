import torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange
from .patch_embed import PatchEmbed3D
from .lora import LoRALinear
from .moe import MoELayer

class Attention(nn.Module):
    def __init__(self, dim, heads=4, r=8):
        super().__init__()
        self.heads = heads; self.scale = (dim//heads) ** -0.5
        self.q = LoRALinear(dim, dim, r=r); self.k = LoRALinear(dim, dim, r=r); self.v = LoRALinear(dim, dim, r=r)
        self.o = LoRALinear(dim, dim, r=r)
    def forward(self, x):  # (B,T,dim)
        B,T,C = x.shape; h=self.heads
        q = rearrange(self.q(x), 'b t (h c) -> b h t c', h=h)
        k = rearrange(self.k(x), 'b t (h c) -> b h t c', h=h)
        v = rearrange(self.v(x), 'b t (h c) -> b h t c', h=h)
        attn = (q @ k.transpose(-1,-2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b h t c -> b t (h c)')
        return self.o(out)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, r=8, num_experts=4, top_k=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, heads=heads, r=r)
        self.norm2 = nn.LayerNorm(dim)
        self.moe   = MoELayer(dim, dim*mlp_ratio, num_experts=num_experts, top_k=top_k)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x

class Encoder3D(nn.Module):
    def __init__(self, in_ch=4, embed_dim=64, depth=(1,1,2,1), num_heads=(2,4,4,8), r=8, num_experts=4, top_k=2):
        super().__init__()
        self.patch = nn.ModuleList([PatchEmbed3D(1, embed_dim, patch_size=4) for _ in range(in_ch)])
        dims = [embed_dim*in_ch, embed_dim*in_ch*2, embed_dim*in_ch*4, embed_dim*in_ch*8]
        self.proj = nn.ModuleList([nn.Conv3d(dims[i], dims[i+1], 2, 2) for i in range(3)])  # downsample
        self.blocks = nn.ModuleList()
        for stage, d in enumerate(depth):
            for _ in range(d):
                self.blocks.append(Block(dims[stage], num_heads[stage], r=r, num_experts=num_experts, top_k=top_k))
            if stage<3:
                self.blocks.append(nn.Identity())  # stage delimiter
        self.dims = dims
    def forward(self, x):  # x: (B,C,D,H,W)
        B,C,D,H,W = x.shape
        feats = [self.patch[i](x[:,i:i+1]) for i in range(C)]
        z = torch.cat(feats, dim=1)  # (B, C*embed, D/4,H/4,W/4)
        outs = []
        t = z.flatten(2).transpose(1,2)  # (B,T,C)
        stage=0
        for blk in self.blocks:
            if isinstance(blk, nn.Identity):
                outs.append(z)
                if stage<3:
                    z = self.proj[stage](z)
                    t = z.flatten(2).transpose(1,2)
                    stage+=1
                continue
            t = blk(t)
            z = t.transpose(1,2).reshape(B, self.dims[stage], z.shape[2], z.shape[3], z.shape[4])
        outs.append(z)
        s8, s16, s32, s64 = outs  # four scales
        return s8, s16, s32
