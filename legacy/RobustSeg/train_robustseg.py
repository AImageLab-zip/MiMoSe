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
from tensorboardX import SummaryWriter

from RobustSeg import RobustSeg
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from utils.criterions import softmax_dice_loss, cross_loss, compute_KLD, dice_loss, softmax_weighted_loss, kl_loss, softmax_loss
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader, LR_Scheduler_polinomial
from utils.predict import AverageMeter, test_softmax_RS
from utils.utils import setup_seed
import wandb
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='/home/SHARE2/ZXY/RFNet/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='/home/SHARE2/ZXY/RF2Robust/results', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--drop_rate_value', default=0.0, type=float)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--scale', default=8, type=int)
parser.add_argument('--debug', action='store_true', default=False)


path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))
val_check = [10, 15, 20, 25, 30, 40, 50, 60, 70, 90, 110, 130, 150, 175, 200, 225, 250, 260, 270, 275, 280, 285, 290, 295, 300] 

###modality missing mask  
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1ce', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

def main():
    setup_seed(args.seed)

    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)
    
    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'BraTS23_RobustSeg_epoch{args.num_epochs}_lr{args.lr}_jobid{slurm_job_id}'
    if not args.debug:
        wandb.init(
            project="MissingBench",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "RobustSeg",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "datapath": args.datapath,
            }
        )
    
    ##########setting models
    num_cls = 4
    model = RobustSeg(num_cls=num_cls)
    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ########## Setting learning scheduler and optimizer
    lr_scheduler = LR_Scheduler_polinomial(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    mse_loss = torch.nn.MSELoss()

    ########## Setting data
    train_file = 'datalist/train.txt'
    test_file = 'datalist/test15splits.csv'
    val_file = 'datalist/val15splits.csv'
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
    iter_per_epoch = len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0

    ##########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        step_lr = lr_scheduler(optimizer, epoch)
        #writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()

        cross_loss_epoch = 0.0
        dice_loss_epoch = 0.0
        reconstruction_loss_epoch = 0.0
        kl_loss_epoch = 0.0
        #l2_loss_epoch = 0.0
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

            outputs = model(x, mask)

            # segmentation loss
            loss_cross = softmax_weighted_loss(outputs['seg_pred'], target, num_cls=num_cls)
            loss_dice = dice_loss(outputs['seg_pred'], target, num_cls=num_cls)

            cross_loss_epoch += loss_cross
            dice_loss_epoch += loss_dice

            # reconstruction loss
            rec_loss =  torch.zeros(1).cuda().float()
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,0:1,:,:,:] - outputs['reconstruct_flair']))
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,1:2,:,:,:] - outputs['reconstruct_t1c']))
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,2:3,:,:,:] - outputs['reconstruct_t1']))
            rec_loss = rec_loss + torch.mean(torch.abs(x[:,3:4,:,:,:] - outputs['reconstruct_t2']))
            reconstruction_loss_epoch += rec_loss

            # kl loss (p(a_i) || N(0, 1))
            KL_loss =  torch.zeros(1).cuda().float()
            KL_loss = KL_loss + kl_loss(outputs['mu_flair'], torch.log(torch.square(outputs['sigma_flair'])))
            KL_loss = KL_loss + kl_loss(outputs['mu_t1c'], torch.log(torch.square(outputs['sigma_t1c'])))
            KL_loss = KL_loss + kl_loss(outputs['mu_t1'], torch.log(torch.square(outputs['sigma_t1'])))
            KL_loss = KL_loss + kl_loss(outputs['mu_t2'], torch.log(torch.square(outputs['sigma_t2'])))
            kl_loss_epoch += KL_loss

            #loss  = loss_cross + loss_dice + l2_loss + 0.1*rec_loss + 0.1*KL_loss
            loss  = loss_cross + loss_dice + 0.1 * rec_loss + 0.1 * KL_loss
            loss_epoch += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('ce_loss', loss_cross.item(), global_step=step)
            writer.add_scalar('dice_loss', loss_dice.item(), global_step=step)
            #writer.add_scalar('l2_loss', l2_loss.item(), global_step=step)
            writer.add_scalar('rec_loss', 0.1*rec_loss.item(), global_step=step)
            writer.add_scalar('kl_loss', 0.1*KL_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'ce_loss:{:.4f},'.format(loss_cross.item())
            msg += 'dice_loss:{:.4f},'.format(loss_dice.item())
            #msg += 'l2_loss:{:.4f},'.format(l2_loss.item())
            msg += 'rec_loss:{:.4f}, kl_loss:{:.4f},'.format(rec_loss.item(), KL_loss.item())
            logging.info(msg)

            if args.debug:
                break

        logging.info('train time per epoch: {}'.format(time.time() - b))

        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_kl": kl_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_reconstruction": reconstruction_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_cross": cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_dice": dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                #"train/l2_loss": l2_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/learning_rate": step_lr,
            })
        
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            },
            file_name)
        
        ########## validation and test
        if (epoch+1) in val_check or args.debug:
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax_RS(
                    val_loader,
                    model,
                    dataname = args.dataname,
                    name='RS')
        
            val_WT, val_TC, val_ET, val_ETpp = dice_score #validate(model, val_loader)
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
                    'val_Dice_best': val_Dice_best,
                    },
                    file_name)
                
            print('testing ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax_RS(
                    test_loader,
                    model,
                    dataname = args.dataname,
                    name='RS')
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

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

if __name__ == '__main__':
    main()
