import os, random, numpy as np, torch
from pathlib import Path

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def to_device(*xs, device=None):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ys = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            ys.append([t.to(device) for t in x])
        elif hasattr(x, 'to'):
            ys.append(x.to(device))
        else:
            ys.append(x)
    return ys if len(ys)>1 else ys[0]

def load_yaml(path):
    import yaml, io
    with open(path, 'r') as f: return yaml.safe_load(f)

def save_checkpoint(path, model, optimizer, epoch, best_metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best': best_metrics}, path)
