import torch, torch.nn as nn, torch.nn.functional as F
from .lora import LoRALinear

class ExpertFFN(nn.Module):
    def __init__(self, dim, hidden, r=8):
        super().__init__()
        self.fc1 = LoRALinear(dim, hidden, r=r)
        self.act = nn.GELU()
        self.fc2 = LoRALinear(hidden, dim, r=r)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class MoELayer(nn.Module):
    def __init__(self, dim, hidden, num_experts=4, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(dim, hidden) for _ in range(num_experts)])
        self.router  = nn.Linear(dim, num_experts)
        self.top_k = top_k
    def forward(self, x):  # x: (B,T,dim)
        logits = self.router(x)                  # (B,T,E)
        topv, topi = logits.topk(self.top_k, dim=-1)
        weights = topv.softmax(dim=-1)          # (B,T,k)
        y = 0.0
        for j in range(self.top_k):
            idx = topi[..., j]                  # (B,T)
            out = 0.0
            for e, expert in enumerate(self.experts):
                mask = (idx==e).unsqueeze(-1)
                if mask.any():
                    out = out + torch.where(mask, expert(x), torch.zeros_like(x))
            y = y + weights[..., j].unsqueeze(-1) * out
        return y
