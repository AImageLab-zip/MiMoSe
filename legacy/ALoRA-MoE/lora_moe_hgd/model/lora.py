import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=1.0, bias=True, freeze_main=False):
        super().__init__()
        self.main = nn.Linear(in_f, out_f, bias=bias)
        if freeze_main:
            for p in self.main.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(out_f, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, in_f))
        self.scaling = alpha / max(1,r)
    def forward(self, x):
        return self.main(x) + (x @ self.B.t() @ self.A.t()) * self.scaling
