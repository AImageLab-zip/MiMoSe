# coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict
import setproctitle
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.utils import setup_seed
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from model.net import Model
from utils import criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='BRATS2020_Training_none_npy', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--user', default='user of name', type=str)
parser.add_argument('--savepath', default='./BraTS2020/split', type=str)
parser.add_argument('--savevisualpath', default='BraTS2020/visual', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--load', default=False, type=bool)  # c加载模型
parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2', 'flairt1cet1t2']
print(masks_torch.int())

#to be changed if you change num_epochs (this is for num_epochs = 500)
val_check = [1, 10, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 730, 760, 790, 820, 850, 880, 900, 920, 940, 960, 970, 980, 985, 990, 995, 998, 999, 1000] 

def main():
    ##########setting seed
    setup_seed(args.seed)
    
    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'{args.dataname}_srmnet_epoch{args.num_epochs}_iter{args.iter_per_epoch}_jobid{slurm_job_id}'
    if not args.debug:
        wandb.init(
            project="MissingBench",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "SRMNet",
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
        print('dataset is error')
        exit(0)
    model = Model(num_cls=num_cls)
    # print(model)
    model = torch.nn.DataParallel(model).cuda()

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ########## Setting data
    if args.dataname in ['BRATS2023']:
        train_file = 'datalist/train2.txt'
        test_file = 'datalist/test15splits.csv'
        val_file = 'datalist/val15splits.csv'
    elif args.dataname == 'BRATS2018':
        test_file = 'datalist18/Brats18_test15splits.csv'
        val_file = 'datalist18/Brats18_val15splits.csv'
        train_file = 'datalist18/Brats18_train3.txt'
    else:
        raise NotImplementedError

    logging.info(str(args))
        
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, val_file=val_file)
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


    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = len(train_loader) if args.iter_per_epoch == -1 else args.iter_per_epoch
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0
    
    # Load model only when explicitly requested and checkpoint path is valid.
    resume_path = args.resume
    if args.load and resume_path and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1
    elif args.load and resume_path:
        logging.warning('cannot resume from %s because file does not exist', resume_path)
    
    for epoch in range(start_epoch, args.num_epochs):
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.num_epochs))
        setproctitle.setproctitle('{}'.format(args.user))
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        
        model.train()
        model.module.is_training=True
        
        #loss per epoch da loggare
        pred_cross_loss_epoch = 0.0
        pred_dice_loss_epoch = 0.0
        preds_cross_loss_epoch = 0.0
        preds_dice_loss_epoch = 0.0
        rec_loss_epoch = 0.0
        loss_epoch = 0.0
        
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch

            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)

            flair = x[:, 0:1, :, :, :]
            t1ce = x[:, 1:2, :, :, :]
            t1 = x[:, 2:3, :, :, :]
            t2 = x[:, 3:4, :, :, :]

            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            #model.module.is_training = True
            pred, preds, recs = model(x, mask)

            ###Loss compute
            pred_cross_loss = criterions.softmax_weighted_loss(pred, target, num_cls=num_cls)
            pred_dice_loss = criterions.dice_loss(pred, target, num_cls=num_cls)
            pred_loss = pred_cross_loss + pred_dice_loss
            
            pred_cross_loss_epoch += pred_cross_loss
            pred_dice_loss_epoch += pred_dice_loss

            preds_cross_loss = torch.zeros(1).cuda().float()
            preds_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in preds:
                preds_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                preds_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            preds_loss = preds_cross_loss + preds_dice_loss
            
            preds_cross_loss_epoch += preds_cross_loss
            preds_dice_loss_epoch += preds_dice_loss

            rec_loss = criterions.l1loss(flair, recs[0]) + criterions.l1loss(t1ce, recs[1]) + criterions.l1loss(t1, recs[2]) + criterions.l1loss(t2, recs[3])
            rec_loss_epoch += rec_loss

            loss = pred_loss + preds_loss + 0.1 * rec_loss
            loss_epoch += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('pred_cross_loss', pred_cross_loss.item(), global_step=step)
            writer.add_scalar('pred_dice_loss', pred_dice_loss.item(), global_step=step)
            writer.add_scalar('preds_cross_loss', preds_cross_loss.item(), global_step=step)
            writer.add_scalar('preds_dice_loss', preds_dice_loss.item(), global_step=step)
            writer.add_scalar('rec_loss', rec_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            msg += 'predcross:{:.4f}, preddice:{:.4f},'.format(pred_cross_loss.item(), pred_dice_loss.item())
            msg += 'predscross:{:.4f}, predsdice:{:.4f},'.format(preds_cross_loss.item(), preds_dice_loss.item())
            msg += 'recloss:{:.4f},'.format(rec_loss.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))
        
        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/pred_cross_loss": pred_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/pred_dice_loss": pred_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/preds_cross_loss": preds_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/preds_dice_loss": preds_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/rec_loss": rec_loss_epoch.cpu().detach().item() / iter_per_epoch,
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

        if epoch+1 in val_check:
            with torch.no_grad():
                dice_score = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname)
            
            val_WT, val_TC, val_ET, val_ETpp = dice_score
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
                
            if val_check.index(epoch+1) % 5 == 0:
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

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

if __name__ == '__main__':
    main()

