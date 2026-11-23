import torch

def dice_score(logits, target, class_idx=1, eps=1e-6):
    # logits: (B,C,D,H,W), target: (B,1,D,H,W) integer {0..C-1} or multi-hot
    if logits.shape[1] == 1:
        pred = (torch.sigmoid(logits)>0.5).float()
        tgt = (target>0.5).float()
    else:
        pred_cls = logits.argmax(dim=1, keepdim=True)  # (B,1,D,H,W)
        pred = (pred_cls==class_idx).float()
        tgt  = (target==class_idx).float()
    inter = (pred*tgt).sum(dim=(2,3,4))
    denom = pred.sum(dim=(2,3,4)) + tgt.sum(dim=(2,3,4)) + eps
    return (2*inter/denom).mean().item()
