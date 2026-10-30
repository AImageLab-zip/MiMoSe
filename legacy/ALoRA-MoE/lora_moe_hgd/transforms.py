import torch, numpy as np
import torch.nn.functional as F

def zscore(x, eps=1e-6):
    mu = x.mean(); sd = x.std()
    return (x - mu) / (sd + eps)

def random_flip(x, y):
    # x: (C,D,H,W), y: (1,D,H,W)
    if torch.rand(1)<0.5: x = x.flip(-1); y = y.flip(-1)
    if torch.rand(1)<0.5: x = x.flip(-2); y = y.flip(-2)
    if torch.rand(1)<0.5: x = x.flip(-3); y = y.flip(-3)
    return x, y
