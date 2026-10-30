import argparse, os, torch, numpy as np
from pathlib import Path
from lora_moe_hgd.model.seg_model import SegModel
from lora_moe_hgd.utils import load_yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--config', type=str, default='configs/brats2018.yaml')
    ap.add_argument('--in-path', type=str, required=False, help='Path with 4 npy volumes or NIfTI (toy path used).')
    ap.add_argument('--out-path', type=str, required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SegModel(in_ch=len(cfg['data']['modalities']),
                     embed_dim=cfg['model']['embed_dim'],
                     depth=tuple(cfg['model']['depth']),
                     num_heads=tuple(cfg['model']['num_heads']),
                     lora_r=cfg['model']['lora_r'],
                     num_experts=cfg['model']['moe']['num_experts'],
                     top_k=cfg['model']['moe']['top_k'],
                     out_ch=cfg['model']['out_channels']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()

    # For demo, read a toy case from data/..../val
    root = Path(cfg['data']['root'])/'val'
    pid = sorted([p.stem for p in root.glob('*_x.npy')])[0]
    x = torch.from_numpy(np.load(root/f"{pid}_x.npy")).float().unsqueeze(0).to(device)  # (1,C,D,H,W)

    with torch.no_grad():
        logits = model(x)
        preds  = logits.argmax(1).cpu().numpy().astype('uint8')  # (1,D,H,W)

    outp = Path(args.out_path); outp.mkdir(parents=True, exist_ok=True)
    np.save(outp/'pred.npy', preds)
    print('Saved:', outp/'pred.npy')

if __name__ == "__main__":
    main()
