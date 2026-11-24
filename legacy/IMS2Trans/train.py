# coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.random_seed import setup_seed
import ims2trans
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax
path = os.path.dirname(__file__)
import wandb 
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()

parser.add_argument(
    "-batch_size", "--batch_size", default=1, type=int, help="Batch size"
)
parser.add_argument("--datapath", default=None, type=str)
parser.add_argument("--dataname", default="BRATS2018", type=str)
parser.add_argument("--savepath", default=None, type=str)
parser.add_argument("--resume", default=None, type=str)
parser.add_argument("--pretrain", default=None, type=str)
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--num_epochs", default=1000, type=int)
parser.add_argument("--iter_per_epoch", default=150, type=int)
parser.add_argument("--region_fusion_start_epoch", default=0, type=int)
parser.add_argument("--res", default=0, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument("--use_checkpoint", action="store_true", help="Enable gradient checkpointing")
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, "training")
args.train_transforms = "Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])"
args.test_transforms = "Compose([NumpyType((np.float32, np.int64)),])"

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
train_log_dir = "summary/train/"
writer = SummaryWriter(os.path.join(args.savepath, train_log_dir))

###modality missing mask
masks = [
    [False, False, False, True],
    [False, True, False, False],
    [False, False, True, False],
    [True, False, False, False],
    [False, True, False, True],
    [False, True, True, False],
    [True, False, True, False],
    [False, False, True, True],
    [True, False, False, True],
    [True, True, False, False],
    [True, True, True, False],
    [True, False, True, True],
    [True, True, False, True],
    [False, True, True, True],
    [True, True, True, True],
]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = [
    "t2",
    "t1c",
    "t1",
    "flair",
    "t1cet2",
    "t1cet1",
    "flairt1",
    "t1t2",
    "flairt2",
    "flairt1ce",
    "flairt1cet1",
    "flairt1t2",
    "flairt1cet2",
    "t1cet1t2",
    "flairt1cet1t2",
]
print(masks_torch.int())

val_check = [50, 70, 90, 110, 130, 150, 200, 300, 400, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 800, 825, 850, 900, 910, 920, 930, 940, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000] 
print(f"Validation checks: {val_check}")


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    Z = size[4]

    cut_rat = np.cbrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cut_z = np.int32(Z * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(Z)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_z // 2, 0, Z)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_z // 2, 0, Z)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2


def main():
    ##########setting seed
    setup_seed(args.seed)

    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)
    
        ##########init wandb
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        wandb_name_and_id = f'{args.dataname}_Scalable_Swin_Transformer_Network_epoch{args.num_epochs}_iter{args.iter_per_epoch}_jobid{slurm_job_id}'
        if not args.debug:
            wandb.init(
                project="MissingBench",
                name=wandb_name_and_id,
                id=wandb_name_and_id,
                resume="allow",
                config={
                    "architecture": "IMS2Trans",
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "iter_per_epoch": args.iter_per_epoch,
                    "num_epochs": args.num_epochs,
                    "datapath": args.datapath,
                    "region_fusion_start_epoch": args.region_fusion_start_epoch,
                }
            )

    ##########setting models
    if args.dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)

    model = ims2trans.Model(num_cls=num_cls, use_checkpoint=args.use_checkpoint)
    print(model)
    model = torch.nn.DataParallel(model).cuda()
    # Totale parametri: 4,685,358
    # Parametri allenabili: 4,685,358

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [
        {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
    ]
    optimizer = torch.optim.Adam(
        train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True
    )
    scaler = GradScaler()

    ########## Setting data
    if args.dataname in ['BRATS2023']:
        train_file = 'datalist/train2.txt'
        test_file = 'datalist/test15splits.csv'
        val_file = 'datalist/val15splits.csv'
    elif args.dataname == 'BRATS2018':
        test_file = 'datalist/Brats18_test15splits.csv'
        val_file = 'datalist/Brats18_val15splits.csv'
        train_file = 'datalist/Brats18_train3.txt'
    else:
        raise NotImplementedError

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, num_cls=num_cls, val_file=val_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0
    cont_loss = criterions.ContrastiveLoss()
    dis_lambda = 0.1  # FDC

    ##########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        scaler.load_state_dict(checkpoint['scaler_dict'])
        start_epoch = checkpoint['epoch'] + 1

    his_x = None
    his_target = None
    his_mask = None

    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar("lr", step_lr, global_step=(epoch + 1))
        b = time.time()
        model.train()
        model.module.is_training = True
        print("checkpoint enabled:", model.module.swinEncoder.layers1[0].use_checkpoint)

        prm_cross_loss_epoch = 0.0
        prm_dice_loss_epoch = 0.0
        fuse_cross_loss_epoch = 0.0
        fuse_dice_loss_epoch = 0.0
        dis_fdc_loss_epoch = 0.0
        dis_loss_epoch = 0.0
        loss_epoch = 0.0

        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            
            x, target, mask = data[:3] # x=(B, M=4, 128, 128, 128), target = (B, C, 128, 128, 128), mask = (B, 4)
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            # generate mixed sample: CutMix 3D
            if his_x is None and his_target is None and his_mask is None:
                his_x = x
                his_target = target
                his_mask = mask

            new_x = torch.cat([his_x, x], dim=0)
            new_target = torch.cat([his_target, target], dim=0)

            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(new_x.size(0), device=new_x.device)
            target_a = new_target
            target_b = new_target[rand_index]
            bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox(new_x.size(), lam)
            new_x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = new_x[
                rand_index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1)
                * (bby2 - bby1)
                * (bbz2 - bbz1)
                / (new_x.size()[-1] * new_x.size()[-2] * new_x.size()[-3])
            )
            his_x = x
            his_target = target

            new_x1, new_x2 = torch.chunk(new_x, 2, dim=0)

            # compute output
            with autocast():
                fuse_pred1, dis_preds1, prm_preds1 = model(new_x1, his_mask)
                fuse_pred2, dis_preds2, prm_preds2 = model(new_x2, mask)
                fuse_pred = torch.cat((fuse_pred1, fuse_pred2), dim=0)

                dis_preds = []
                for j in range(len(dis_preds1) - 1):
                    dis_preds.append(torch.cat((dis_preds1[j], dis_preds2[j]), dim=0))
                dis_target = torch.cat((dis_preds1[-1], dis_preds2[-1]), dim=0)
                
                prm_preds = []
                for j in range(len(prm_preds1)):
                    prm_preds.append(torch.cat((prm_preds1[j], prm_preds2[j]), dim=0))

                his_mask = mask

                ###Loss compute
                device = fuse_pred.device
                dtype  = fuse_pred.dtype
                dis_fdc_loss  = torch.zeros(1, device=device, dtype=dtype)
                prm_cross_loss = torch.zeros(1, device=device, dtype=dtype)
                prm_dice_loss  = torch.zeros(1, device=device, dtype=dtype)

                fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target_a, num_cls=num_cls) * lam + \
                                criterions.softmax_weighted_loss(fuse_pred, target_b, num_cls=num_cls) * (1.0 - lam)

                fuse_dice_loss  = criterions.dice_loss(fuse_pred, target_a, num_cls=num_cls) * lam + \
                                criterions.dice_loss(fuse_pred, target_b, num_cls=num_cls) * (1.0 - lam)

                fuse_loss = fuse_cross_loss + fuse_dice_loss

                #dis_fdc_loss = torch.zeros(1).cuda().float()
                for dis_pred in dis_preds:
                    dis_fdc_loss += cont_loss(dis_pred, dis_target)
                dis_loss = dis_lambda * dis_fdc_loss

                #prm_cross_loss = torch.zeros(1).cuda().float()
                #prm_dice_loss = torch.zeros(1).cuda().float()
                for prm_pred in prm_preds:
                    prm_cross_loss += criterions.softmax_weighted_loss(
                        prm_pred, target_a, num_cls=num_cls
                    ) * lam + criterions.softmax_weighted_loss(
                        prm_pred, target_b, num_cls=num_cls
                    ) * (
                        1.0 - lam
                    )
                    prm_dice_loss += criterions.dice_loss(
                        prm_pred, target_a, num_cls=num_cls
                    ) * lam + criterions.dice_loss(prm_pred, target_b, num_cls=num_cls) * (
                        1.0 - lam
                    )
                prm_loss = prm_cross_loss + prm_dice_loss

                ### total segmentation loss
                if epoch < args.region_fusion_start_epoch:
                    loss = fuse_loss * 0.0 + dis_loss + prm_loss
                else:
                    loss = fuse_loss + dis_loss + prm_loss

            # --- backprop (AMP) ---
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- logging accumulators ---
            loss_epoch += loss.item()
            fuse_cross_loss_epoch += fuse_cross_loss.item()
            fuse_dice_loss_epoch  += fuse_dice_loss.item()
            dis_fdc_loss_epoch    += dis_fdc_loss.item()
            dis_loss_epoch        += dis_loss.item()
            prm_cross_loss_epoch  += prm_cross_loss.item()
            prm_dice_loss_epoch   += prm_dice_loss.item()

            ###log
            writer.add_scalar("loss", loss.item(), global_step=step)
            writer.add_scalar(
                "fuse_cross_loss", fuse_cross_loss.item(), global_step=step
            )
            writer.add_scalar("fuse_dice_loss", fuse_dice_loss.item(), global_step=step)
            writer.add_scalar("dis_fdc_loss", dis_fdc_loss.item(), global_step=step)
            writer.add_scalar("dis_loss", dis_loss.item(), global_step=step)
            writer.add_scalar("prm_cross_loss", prm_cross_loss.item(), global_step=step)
            writer.add_scalar("prm_dice_loss", prm_dice_loss.item(), global_step=step)

            msg = "Epoch {}/{}, Iter {}/{}, Loss {:.4f}, ".format(
                (epoch + 1), args.num_epochs, (i + 1), iter_per_epoch, loss.item()
            )
            msg += "fusecross:{:.4f}, fusedice:{:.4f},".format(
                fuse_cross_loss.item(), fuse_dice_loss.item()
            )
            msg += "dis_fdc_loss:{:.4f}, dis_loss:{:.4f},".format(
                dis_fdc_loss.item(), dis_loss.item()
            )
            msg += "prmcross:{:.4f}, prmdice:{:.4f},".format(
                prm_cross_loss.item(), prm_dice_loss.item()
            )
            logging.info(msg)
        
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch / iter_per_epoch,
                "train/fusecross": fuse_cross_loss_epoch / iter_per_epoch,
                "train/fusedice": fuse_dice_loss_epoch / iter_per_epoch,
                "train/dis_fdc_loss": dis_fdc_loss_epoch / iter_per_epoch,
                "train/dis_loss": dis_loss_epoch / iter_per_epoch,
                "train/prmcross": prm_cross_loss_epoch / iter_per_epoch,
                "train/prmdice": prm_dice_loss_epoch / iter_per_epoch,
                "train/learning_rate": step_lr,
            })

        ########## model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scaler_dict': scaler.state_dict(),
            'val_Dice_best': val_Dice_best,
            },
            file_name)

        ########## validation and test
        if epoch+1 in val_check:
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname)
        
            val_WT, val_TC, val_ET, val_ETpp = dice_score 
            logging.info('Validate epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}, loss = {:.2}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), seg_loss.cpu().item()))
            val_dice = (val_ET + val_WT + val_TC)/3
            if not args.debug:
                wandb.log({
                    "val/epoch":epoch,
                    "val/val_ET_Dice": val_ET.item(),
                    "val/val_ETpp_Dice": val_ETpp.item(),
                    "val/val_WT_Dice": val_WT.item(),
                    "val/val_TC_Dice": val_TC.item(),
                    "val/val_Dice": val_dice.item(), 
                    "val/seg_loss": seg_loss.cpu().item(),       
                })
            
            if val_dice > val_Dice_best:
                val_Dice_best = val_dice.item()
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'scaler_dict': scaler.state_dict(),
                    'val_Dice_best': val_Dice_best,
                    },
                    file_name)
                
            print('testing ...')
            test_score = AverageMeter()
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    test_loader,
                    model,
                    dataname = args.dataname)
            test_WT, test_TC, test_ET, test_ETpp = dice_score   
            logging.info('Testing epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ET_postpro = {:.2}'.format(epoch, test_WT.item(), test_TC.item(), test_ET.item(), test_ETpp.item()))
            test_dice = (test_ET + test_WT + test_TC)/3
            if not args.debug:
                wandb.log({
                    "test/epoch":epoch,
                    "test/test_WT_Dice": test_WT.item(),
                    "test/test_TC_Dice": test_TC.item(),
                    "test/test_ET_Dice": test_ET.item(),
                    "test/test_ETpp": test_ETpp.item(),
                    "test/test_Dice": test_dice.item(),  
                    "test/seg_loss": seg_loss.cpu().item(),   
                })

            model.train()
            model.module.is_training=True

    msg = "total time: {:.4f} hours".format((time.time() - start) / 3600)
    logging.info(msg)

if __name__ == "__main__":
    main()
