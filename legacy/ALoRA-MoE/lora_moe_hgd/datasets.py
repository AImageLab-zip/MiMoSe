import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .transforms import zscore, random_flip

def simulate_missing_modalities(x, drop_p=0.3):
    # x: (C,D,H,W), zero out random subset (at least one channel kept)
    C = x.shape[0]
    mask = torch.ones(C, dtype=torch.bool)
    for c in range(C):
        if torch.rand(1).item() < drop_p: mask[c]=False
    if mask.sum()==0: mask[torch.randint(0,C,(1,))]=True
    x[~mask] = 0.0
    return x, mask.float()

class ToyBraTSDataset(Dataset):
    # Synthetic 3D dataset (C=4) writing/reading small .npy volumes for fast tests.
    # Replace this with a NIfTI reader for real BraTS.
    def __init__(self, root, split="train", patch_size=(64,64,64), drop_p=0.2):
        self.root = Path(root)/split
        self.ids  = sorted([p.stem for p in self.root.glob("*_x.npy")])
        self.patch_size = patch_size
        self.drop_p = drop_p

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        x = torch.from_numpy(np.load(self.root/f"{pid}_x.npy")).float()  # (C,D,H,W)
        y = torch.from_numpy(np.load(self.root/f"{pid}_y.npy")).long()   # (1,D,H,W) classes {0..3}

        # z-score per-channel
        for c in range(x.shape[0]):
            xc = x[c]; x[c] = (xc - xc.mean()) / (xc.std()+1e-6)

        # random flips
        x, y = random_flip(x, y)

        # simulate missing modalities
        x, avail = simulate_missing_modalities(x, self.drop_p)

        return x, y, avail

def build_loaders(cfg, stage_idx=0, num_workers=2):
    root = cfg['data']['root']
    ps   = tuple(cfg['data']['patch_size'])
    drops= cfg['data']['missing_drop_p']
    drop_p = float(drops[min(stage_idx, len(drops)-1)])
    train_set = ToyBraTSDataset(root, split="train", patch_size=ps, drop_p=drop_p)
    val_set   = ToyBraTSDataset(root, split="val",   patch_size=ps, drop_p=0.0)
    train_loader = DataLoader(train_set, batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
