import torch, torch.nn as nn, torch.nn.functional as F

class HGDDecoder(nn.Module):
    def __init__(self, in_channels=(256,512,1024), mid=256, w=32, out_ch=4):
        super().__init__()
        c8, c16, c32 = in_channels
        self.c8  = nn.Conv3d(c8,  512, 1)
        self.c16 = nn.Conv3d(c16, 512, 1)
        self.c32 = nn.Conv3d(c32, 512, 1)

        self.bm = nn.Conv3d(512*3, mid, 1)   # base map
        self.sm = nn.Conv3d(512*3, w,   1)   # spatial weights

        self.g1 = nn.Conv3d(512*3, mid, 1)  # guidance
        self.g2 = nn.Conv3d(mid, w, 1)      # codeword weights

        self.head = nn.Conv3d(mid*2, out_ch, 1)

    def _fuse(self, g8, g16, g32, to='s32'):
        if to=='s32':
            g8d  = torch.nn.functional.interpolate(g8,  size=g32.shape[-3:], mode='trilinear', align_corners=False)
            g16d = torch.nn.functional.interpolate(g16, size=g32.shape[-3:], mode='trilinear', align_corners=False)
            return torch.cat([g8d, g16d, g32], dim=1)
        else:
            g16u = torch.nn.functional.interpolate(g16, size=g8.shape[-3:], mode='trilinear', align_corners=False)
            g32u = torch.nn.functional.interpolate(g32, size=g8.shape[-3:], mode='trilinear', align_corners=False)
            return torch.cat([g8, g16u, g32u], dim=1)

    def forward(self, s8, s16, s32):
        g8, g16, g32 = self.c8(s8), self.c16(s16), self.c32(s32)
        c32 = self._fuse(g8, g16, g32, to='s32')
        c8  = self._fuse(g8, g16, g32, to='s8')

        BM = self.bm(c32)                   # (B,mid, d32,h32,w32)
        SM = self.sm(c32)                   # (B,w,   d32,h32,w32)
        SM = SM.flatten(2).softmax(-1)      # spatial softmax
        BMf= BM.flatten(2).transpose(1,2)   # (B, N32, mid)
        CW = torch.einsum('bwn,bnq->bwq', SM, BMf)  # (B,w,mid)

        FVg = self.g1(c8)                   # (B,mid, d8,h8,w8)
        BMmean = BM.mean(dim=(2,3,4), keepdim=True)
        FVg = FVg + BMmean
        CWw = self.g2(FVg).flatten(2)       # (B,w,N8)
        CWw = CWw.softmax(1)
        FV8 = torch.einsum('bwq,bwk->bqk', CW, CWw).reshape(FVg.shape)
        Fcat = torch.cat([FV8, FVg], dim=1)
        logits_s8 = self.head(Fcat)
        return torch.nn.functional.interpolate(logits_s8, scale_factor=8, mode='trilinear', align_corners=False)
