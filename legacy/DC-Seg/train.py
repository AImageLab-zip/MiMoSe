#coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from utils.utils import setup_seed
import models
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.datasets_wmh import WMH_loadall_nii, WMH_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--datapath', default='BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='Brats2020', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=20, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--fusion_type', default='RFM', type=str)
parser.add_argument('--use_recon_loss', action='store_true', default=False)
parser.add_argument('--use_reg_loss', action='store_true', default=False)
parser.add_argument('--use_ana_contrastive', action='store_true', default=False)
parser.add_argument('--use_mod_contrastive', action='store_true', default=False)
parser.add_argument('--crop_size', default=80, type=int)
parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
if args.dataname == 'wmh':
    args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
else:
    args.train_transforms = f'Compose([RandCrop3D(({args.crop_size},{args.crop_size},{args.crop_size})), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
if args.dataname == 'wmh':
    masks = [[False, True], [True, False], [True, True]]
    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = ['t1', 'flair', 't1flair']
else:
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
            [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
            [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
            [True, True, True, True]]
    masks_torch = torch.from_numpy(np.array(masks))
    mask_name = ['t2', 't1c', 't1', 'flair', 
                't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                'flairt1cet1t2']
print (masks_torch.int())

#to be changed if you change num_epochs (this is for num_epochs = 500)
val_check = [1, 10, 30, 50, 70, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 380, 400, 410, 420, 430, 440, 450, 460, 470, 475, 480, 485, 490, 495, 498, 499, 500] 

def main():
    ##########setting seed
    setup_seed(args.seed)

    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'{args.dataname}_dcseg_epoch{args.num_epochs}_iter{args.iter_per_epoch}_jobid{slurm_job_id}'
    if not args.debug:
        wandb.init(
            project="MissingBench",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "DC-Seg",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "iter_per_epoch": args.iter_per_epoch,
                "num_epochs": args.num_epochs,
                "datapath": args.datapath,
                "region_fusion_start_epoch": args.region_fusion_start_epoch,
            }
        )

    ##########setting models
    if args.dataname in ['BRATS2023', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    elif args.dataname == 'wmh':
        num_cls = 1
    else:
        print ('dataset is error')
        exit(0)
    if args.dataname == 'wmh':
        model = models.DC_Seg_WMH(num_cls=num_cls, fusion_type=args.fusion_type, activation='sigmoid')
    else:
        model = models.DC_Seg(num_cls=num_cls, fusion_type=args.fusion_type)
    # print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    if args.dataname == 'BRATS2023':
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

    ##########Training setup
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0
    
    ##########Evaluate
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1

    ##########Training
    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()

        model.train()
        model.module.is_training = True

        #loss per epoch da loggare
        prm_cross_loss_epoch = 0.0
        prm_dice_loss_epoch = 0.0
        fuse_cross_loss_epoch = 0.0
        fuse_dice_loss_epoch = 0.0
        sep_cross_loss_epoch = 0.0
        sep_dice_loss_epoch = 0.0
        kl_loss_epoch = 0.0
        anatomical_contrastive_loss_epoch = 0.0
        modality_contrastive_loss_epoch = 0.0
        loss_epoch = 0.0

        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds, recon_out, mu_list, sigma_list, contents, styles = model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss if args.dataname != 'wmh' else fuse_dice_loss

            fuse_cross_loss_epoch += fuse_cross_loss
            fuse_dice_loss_epoch += fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss if args.dataname != 'wmh' else sep_dice_loss

            sep_cross_loss_epoch += sep_cross_loss
            sep_dice_loss_epoch += sep_dice_loss

            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss if args.dataname != 'wmh' else prm_dice_loss

            prm_cross_loss_epoch += prm_cross_loss
            prm_dice_loss_epoch += prm_dice_loss

            use_reg_loss = 1 if args.use_reg_loss else 0
            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss*use_reg_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss*use_reg_loss + prm_loss

            from loss import KL_divergence, Anatomy_Contrastive_Loss, Modality_Contrastive_Loss
            recon_loss = F.mse_loss(recon_out, x)
            kl_loss = 0.0
            num_modal = 4
            for m in range(num_modal):
                kl_loss += KL_divergence(mu_list[m], torch.log(torch.square(sigma_list[m])))

            ana_cl_model = Anatomy_Contrastive_Loss(method='ssim')
            mod_cl_model = Modality_Contrastive_Loss()
            anatomical_contrastive_loss = ana_cl_model(contents)
            modality_contrastive_loss = mod_cl_model(styles)

            kl_loss_epoch += kl_loss
            anatomical_contrastive_loss_epoch += anatomical_contrastive_loss
            modality_contrastive_loss_epoch += modality_contrastive_loss

            alpha = 0.1
            if args.use_recon_loss:
                loss += alpha*recon_loss*4
                # loss += 0.4*recon_loss
            if args.use_ana_contrastive:
                loss += alpha*anatomical_contrastive_loss*4
                # loss += anatomical_contrastive_loss 
            if args.use_mod_contrastive:
                loss += alpha*modality_contrastive_loss*4
                # loss += modality_contrastive_loss
            loss += alpha*kl_loss

            loss_epoch += loss

            ### backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('kl_loss', kl_loss.item(), global_step=step)
            if args.use_recon_loss:
                writer.add_scalar('recon_loss', recon_loss.item(), global_step=step)
            if args.use_ana_contrastive:
                writer.add_scalar('anatomical_contrastive_loss', anatomical_contrastive_loss.item(), global_step=step)
            if args.use_mod_contrastive:
                writer.add_scalar('modality_contrastive_loss', modality_contrastive_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'kl_loss:{:.4f}, '.format(kl_loss.item())
            if args.use_recon_loss:
                msg += 'recon_loss:{:.4f}, '.format(recon_loss.item())
            if args.use_ana_contrastive:
                msg += 'anatomical_contrastive_loss:{:.4f}, '.format(anatomical_contrastive_loss.item())
            if args.use_mod_contrastive:
                msg += 'modality_contrastive_loss:{:.4f}, '.format(modality_contrastive_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/fusecross": fuse_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/fusedice": fuse_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/sepcross": sep_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/sepdice": sep_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/prmcross": prm_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/prmdice": prm_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/kl_loss": kl_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/anatomical_contrastive_loss": anatomical_contrastive_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/modality_contrastive_loss": modality_contrastive_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/learning_rate": step_lr,
            })

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            },
            file_name)
        
        ########## validation and test
        if epoch+1 in val_check:
            print('validate ...')
            with torch.no_grad():
                dice_score = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname)
                
            val_WT, val_TC, val_ET, val_ETpp = dice_score #validate(model, val_loader)
            logging.info('Validate epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))
            val_dice = (val_ET + val_WT + val_TC)/3
            if not args.debug:
                wandb.log({
                    "val/epoch":epoch,
                    "val/val_ET_Dice": val_ET.item(),
                    "val/val_ETpp_Dice": val_ETpp.item(),
                    "val/val_WT_Dice": val_WT.item(),
                    "val/val_TC_Dice": val_TC.item(),
                    "val/val_Dice": val_dice.item(),   
                })

            if val_dice > val_Dice_best:
                val_Dice_best = val_dice.item()
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'val_Dice_best': val_Dice_best,
                    },
                    file_name)
                
            print('testing ...')
            test_score = AverageMeter()
            with torch.no_grad():
                dice_score = test_softmax(
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
                })

            model.train()
            model.module.is_training=True
        

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)


if __name__ == '__main__':
    main()
