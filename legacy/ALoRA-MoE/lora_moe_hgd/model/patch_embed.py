import torch.nn as nn

class PatchEmbed3D(nn.Module):
    def __init__(self, in_ch=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
    def forward(self, x):
        return self.proj(x)
