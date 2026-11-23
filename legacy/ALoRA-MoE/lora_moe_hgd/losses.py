import torch, torch.nn as nn, torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    # pred: (B,1,D,H,W) sigmoid; target: (B,1,D,H,W) in {0,1}
    pred = torch.sigmoid(pred)
    inter = (pred*target).sum(dim=(2,3,4))
    denom = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + eps
    loss = 1 - 2*inter/denom
    return loss.mean()

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
    def forward(self, pred, target):
        return self.w*self.bce(pred, target) + (1-self.w)*dice_loss(pred, target)
