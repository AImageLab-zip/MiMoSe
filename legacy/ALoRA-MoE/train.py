import argparse, os, torch
from pathlib import Path
from lora_moe_hgd.utils import seed_all, load_yaml, save_checkpoint, to_device
from lora_moe_hgd.datasets import build_loaders
from lora_moe_hgd.model.seg_model import SegModel
from lora_moe_hgd.engine import train_one_epoch, validate
from lora_moe_hgd.schedule import curriculum_stage

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/brats2018.yaml')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg['train']['epochs'] = int(args.epochs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(42)

    model = SegModel(in_ch=len(cfg['data']['modalities']),
                     embed_dim=cfg['model']['embed_dim'],
                     depth=tuple(cfg['model']['depth']),
                     num_heads=tuple(cfg['model']['num_heads']),
                     lora_r=cfg['model']['lora_r'],
                     num_experts=cfg['model']['moe']['num_experts'],
                     top_k=cfg['model']['moe']['top_k'],
                     out_ch=cfg['model']['out_channels']).to(device)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    milestones = cfg['train']['curriculum_epochs']
    best = {'WT':0,'TC':0,'ET':0,'mean':0}
    for epoch in range(cfg['train']['epochs']):
        stage = curriculum_stage(epoch, milestones)  # 0,1,2
        tr_loader, va_loader = build_loaders(cfg, stage_idx=stage, num_workers=args.num_workers)
        loss, sec = train_one_epoch(model, tr_loader, opt, device, stage)
        metrics = validate(model, va_loader, device)
        mean_d = (metrics['WT']+metrics['TC']+metrics['ET'])/3.0
        if mean_d > best['mean']:
            best = {'WT':metrics['WT'],'TC':metrics['TC'],'ET':metrics['ET'],'mean':mean_d}
            save_checkpoint("checkpoints/best.pt", model, opt, epoch, best)
        print(f"Epoch {epoch:03d} | stage={stage+1} | loss={loss:.4f} | WT={metrics['WT']:.3f} TC={metrics['TC']:.3f} ET={metrics['ET']:.3f} | best={best['mean']:.3f}")
    print('Done. Best:', best)

if __name__ == "__main__":
    main()
