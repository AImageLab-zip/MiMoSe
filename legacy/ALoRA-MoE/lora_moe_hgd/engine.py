import torch, time
from .losses import DiceBCELoss
from .metrics import dice_score

def train_one_epoch(model, loader, optimizer, device, stage):
    model.train(); t0=time.time()
    loss_fn = DiceBCELoss(0.5).to(device)
    tot=0.0
    for x, y, avail in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)  # (B,4,D,H,W)
        # multi-class: encode per-class targets
        loss = 0.0
        for ci in range(1,4):  # WT(1), TC(2), ET(3) in this toy label
            if stage+1 >= ci:  # curriculum gating
                loss = loss + loss_fn(logits[:,ci:ci+1], (y==ci).float())
        loss.backward(); optimizer.step()
        tot += float(loss.item())
    return tot / max(1,len(loader)), time.time()-t0

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    dices = {'WT':0.0, 'TC':0.0, 'ET':0.0}; n=0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        dices['WT'] += dice_score(logits, y, class_idx=1)
        dices['TC'] += dice_score(logits, y, class_idx=2)
        dices['ET'] += dice_score(logits, y, class_idx=3)
        n+=1
    for k in dices: dices[k] = dices[k]/max(1,n)
    return dices
