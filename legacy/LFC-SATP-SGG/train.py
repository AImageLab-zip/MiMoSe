# coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_nii, Brats_loadall_test_nii,
                               Brats_loadall_val_nii)
from data.transforms import *
# from visualizer import get_local
# get_local.activate()
from models import LFC
from predict import AverageMeter, test_softmax, test_dice_hd95_softmax

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Parser, criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup
# from utils.visualize import visualize_heads
from fvcore.nn import FlopCountAnalysis, parameter_count

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='LFC', type=str)
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--start_epochs', default=0, type=int)

parser.add_argument('--iter_per_epoch', default=100, type=int)
parser.add_argument('--dataname', default='BRATS2018', type=str)

parser.add_argument('--datapath', default='dataset_2018', type=str)
parser.add_argument('--val_metric', default='loss', type=str, help='loss dice')
parser.add_argument('--validation_start_epoch', default=1000, type=int, help='valid')
parser.add_argument('--savepath', default='output_2018_woT1', type=str)
parser.add_argument('--resume', default=None, type=str)

parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1037, type=int)
parser.add_argument('--needvalid', default=False, type=bool)
parser.add_argument('--csvpath', default='output_2018', type=str)

path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

csvpath = args.csvpath
os.makedirs(csvpath, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))
pred_save_dir = os.path.join(args.savepath, 'predictions')
###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
              [True, False, False, False],
              [False, True, False, True], [False, True, True, False], [True, False, True, False],
              [False, False, True, True], [True, False, False, True], [True, True, False, False],
              [True, True, True, False], [True, False, True, True], [True, True, False, True],
              [False, True, True, True],
              [True, True, True, True]]

masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1ce', 't1', 'flair',
             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
             'flairt1cet1t2']

print(masks_torch.int())

# masks_valid = [[False, False, True, False],
#             [False, True, True, False],
#             [True, True, False, True],
#             [True, True, True, True]]
masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
               [True, False, False, False],
               [False, True, False, True], [False, True, True, False], [True, False, True, False],
               [False, False, True, True], [True, False, False, True], [True, True, False, False],
               [True, True, True, False], [True, False, True, True], [True, True, False, True],
               [False, True, True, True],
               [True, True, True, True]]
# t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))
# mask_name_valid = ['t1',
#                 't1cet1',
#                 'flairt1cet2',
#                 'flairt1cet1t2']
mask_name_valid = ['t2', 't1c', 't1', 'flair',
                   't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                   'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                   'flairt1cet1t2']
print(masks_valid_torch.int())
mask_array = np.array(
    [[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
     [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False],
     [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True],
     [True, False, True, True], [False, True, True, True],
     [True, True, True, True]])


def main():
    torch.autograd.set_detect_anomaly(True)
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if args.pred_save_path is not None:
        os.makedirs(args.pred_save_path, exist_ok=True)
    ##########setting models
    if args.dataname in ['BRATS2018', 'BRATS2020', 'BRATS2021']:
        num_cls = 4
    elif args.dataname == 'BRATS2024':
        num_cls = 5
    else:
        print('dataset is error')
        exit(0)

    if args.model == 'fusiontrans':
        model = fusiontrans.Model(num_cls=num_cls)
    elif args.model == 'rfnet':
        model = rfnet.Model(num_cls=num_cls)
    elif args.model == 'mmformer':
        model = mmformer.Model(num_cls=num_cls)
    elif args.model == 'LFC':
        model = LFC.Model(num_cls=num_cls)
    elif args.model == 'cnn':
        model = cnn.Model(num_cls=num_cls)

    # # print (model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
   
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device(f"cuda:{local_rank}")

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # model = torch.nn.DataParallel(model).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model parameter_count:{count_parameters(model)}")
    # mask_idx = np.random.choice(15, 1)
    # mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
    # batchsize = mask.size(0)
    # # all modalities test
    # mask = masks_all_torch.repeat(batchsize, 1)
    # mask = mask[0].repeat(batchsize, 1)  ## to be test
    # input_tensor = torch.randn(1,4, 80, 80, 80)
    # input_tensor=(input_tensor,mask)
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"Model FLOPs: {flops.total()}")

    ##########Setting learning schedule and optimizer

    lr_schedule = LR_Scheduler(args.lr, args.num_epochs, warmup=args.region_fusion_start_epoch, mode='warmuppoly')
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.AdamW(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    ####BRATS2020
    if args.dataname == 'BRATS2020':
        train_file = 'train.txt'
        test_file = 'test1.txt'
        valid_file = 'val.txt'
    elif args.dataname == 'BRATS2024':
        ####BRATS2021
        train_file = 'train.txt'
        test_file = 'test.txt'
        valid_file = 'val.txt'
    elif args.dataname == 'BRATS2021':
        ####BRATS2021
        train_file = 'train.txt'
        test_file = 'test.txt'
        valid_file = 'val.txt'
    elif args.dataname == 'BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train.txt'
        test_file = 'test.txt'
        valid_file = 'val.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                  train_file=train_file, mask_ratio=0.25)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file, )
    valid_set = Brats_loadall_val_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                      train_file=valid_file)
    # train_loader = MultiEpochsDataLoader(
    #     dataset=train_set,
    #     batch_size=args.batch_size,
    #     num_workers=4,
    #     pin_memory=False,
    #     shuffle=True,
    #     worker_init_fn=init_fn)
    # test_loader = MultiEpochsDataLoader(
    #     dataset=test_set,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     pin_memory=False)
    # valid_loader = MultiEpochsDataLoader(
    #     dataset=valid_set,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=init_fn)
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(train_set, shuffle=True)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=init_fn
    )

    valid_sampler = DistributedSampler(valid_set, shuffle=False)
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=init_fn
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # if args.resume is not None:
    #     checkpoint = torch.load(args.resume, weights_only=False)
    #     pretrained_dict = checkpoint['state_dict']
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     # logging.info('pretrained_dict: {}'.format(pretrained_dict))
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    #     logging.info('load ok')
    if args.resume is not None:
        checkpoint = torch.load(args.resume, weights_only=False, map_location=device)  
        pretrained_dict = checkpoint['state_dict']

        model_dict = model.module.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)

        model.module.load_state_dict(model_dict)

        logging.info('load ok')

    #########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    best_metric = float('inf') if args.val_metric == 'loss' else 0.0
    validation_started = False
    # valid_iter = iter(valid_loader)
    
    loss_curve = []
    fuse_cross_curve = []
    fuse_dice_curve = []
    sep_cross_curve = []
    sep_dice_curve = []
    prm_cross_curve = []
    prm_dice_curve = []

    text_align_loss_curve = []
    for epoch in range(args.start_epochs, args.num_epochs):

        train_sampler.set_epoch(epoch)

        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        model.train()  # This sets PyTorch layers (Dropout, etc.) to training mode
        model.is_training = True  # This sets your custom logic to training mode
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
          

            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            # print(f"x device: {x.device}")
            # print(f"target device: {target.device}")
            # print(f"mask device: {mask.device}")
            # x = x.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)

            batchsize = mask.size(0)
            # all modalities test
            # mask = masks_all_torch.repeat(batchsize, 1)
            # mask = mask[0].repeat(batchsize, 1)  ## to be test
            # print(f"mask device: {mask.device}")
            # mask = mask.cuda(non_blocking=True)

            model.is_training = True
            # model.module.is_training = True
            # flops = FlopCountAnalysis(model, (x, mask))
            # print(f"Model FLOPs: {flops.total()}")

            fuse_pred, sep_preds, prm_preds, kl = model(x, mask)

            # fuse_pred, sep_preds, prm_preds= model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)

            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1, device=device).float()
            sep_dice_loss = torch.zeros(1, device=device).float()
            # sep_cross_loss = torch.zeros(1).cuda().float()
            # sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)

            # sep_cross_loss = sep_cross_loss / len(sep_preds)
            # sep_dice_loss  = sep_dice_loss  / len(sep_preds)
            sep_loss = sep_cross_loss + sep_dice_loss

            # kl = kl.to(device)
            kl_loss = criterions.softmax_kl_loss(kl)

            prm_cross_loss = torch.zeros(1, device=device).float()
            prm_dice_loss = torch.zeros(1, device=device).float()
            # prm_cross_loss = torch.zeros(1).cuda().float()
            # prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_pred = prm_pred.to(device)

                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)

            # prm_cross_loss = prm_cross_loss / len(prm_preds)
            # prm_dice_loss  = prm_dice_loss  / len(prm_preds)

            prm_loss = prm_cross_loss + prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0
            else:
                # loss = fuse_loss+ sep_loss
                loss = fuse_loss + sep_loss + prm_loss + kl_loss
            # if epoch < args.region_fusion_start_epoch:
            #     w_fuse, w_sep, w_prm, w_kl = 0.0, 1.0, 0.0, 0.0
            # else:
            #     w_fuse, w_sep, w_prm, w_kl = 1.0, 0.30, 0.30, 0.02
            # loss = w_fuse * fuse_loss + w_sep * sep_loss + w_prm * prm_loss + w_kl * kl_loss

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            # msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
            #                                                       loss.item())
            # msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            # msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            # msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            if local_rank == 0:
                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1),
                                                                      iter_per_epoch, loss.item())
                msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
                msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
                msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
                logging.info(msg)

            loss_curve.append(loss.item())
            fuse_cross_curve.append(fuse_cross_loss.item())
            fuse_dice_curve.append(fuse_dice_loss.item())
            sep_cross_curve.append(sep_cross_loss.item())
            sep_dice_curve.append(sep_dice_loss.item())
            prm_cross_curve.append(prm_cross_loss.item())
            prm_dice_curve.append(prm_dice_loss.item())

            # logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b))

        if epoch >= args.validation_start_epoch - 1:
            if not validation_started:
                logging.info(f'{args.validation_start_epoch} ')
                validation_started = True

            model.eval()
            model.is_training = False
            total_val_loss = 0.0

            with torch.no_grad():
                for val_data in valid_loader:
                    val_x, val_target, val_mask = val_data[:3]
                    val_x = val_x.to(device, non_blocking=True)
                    val_target = val_target.to(device, non_blocking=True)
                    val_mask = val_mask.to(device, non_blocking=True)

                    fuse_pred = model(val_x, val_mask)
                    if isinstance(fuse_pred, tuple): fuse_pred = fuse_pred[0]

                    fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, val_target, num_cls=num_cls)
                    fuse_dice_loss = criterions.dice_loss(fuse_pred, val_target,
                                                          num_cls=num_cls)  
                    fuse_loss = fuse_cross_loss + fuse_dice_loss

                    total_val_loss += fuse_loss.item()

            avg_val_loss = total_val_loss / len(valid_loader)
            # logging.info(f'Epoch {epoch + 1}/{args.num_epochs}, Average Validation Loss: {avg_val_loss:.4f}')
            # writer.add_scalar('validation/loss', avg_val_loss, global_step=(epoch + 1))
            if local_rank == 0:
                logging.info(f'Epoch {epoch + 1}/{args.num_epochs}, Average Validation Loss: {avg_val_loss:.4f}')
                writer.add_scalar('validation/loss', avg_val_loss, global_step=(epoch + 1))

            current_metric = avg_val_loss

            is_best = False
            if args.val_metric == 'loss':
                if current_metric < best_metric:
                    is_best = True

            if is_best:
                if local_rank == 0:
                    logging.info(f'{best_metric:.4f} -> {current_metric:.4f}')
                best_metric = current_metric
                epochs_no_improve = 0

                if local_rank == 0:
                    file_name = os.path.join(ckpts, 'model_best.pth')
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optim_dict': optimizer.state_dict(),
                        'best_metric': best_metric,
                    }, file_name)
                else:
                    epochs_no_improve += 1
                    if local_rank == 0:
                        logging.info(f'{epochs_no_improve}')

        #########model save
        if local_rank == 0:
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

            if (epoch + 1) % 50 == 0 or (epoch >= (args.num_epochs - 10)):
                file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1))
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                },
                    file_name)
    if local_rank == 0:
        msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
        logging.info(msg)

    # ##########Test the last epoch model
    # # writer_visualize = SummaryWriter(log_dir="visualize/result")
    # # visualize_step = 0
    # test_dice_score = AverageMeter()
    # test_hd95_score = AverageMeter()
    # csv_name = os.path.join(csvpath, '{}.csv'.format(args.model))
    # with torch.no_grad():
    #     logging.info('###########test last epoch model###########')
    #     file = open(csv_name, "a+")
    #     csv_writer = csv.writer(file)
    #     # csv_writer.writerow(
    #     #     ['WT Dice', 'TC Dice', 'ET Dice', 'ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95', 'ETPro HD95'])
    #     if args.dataname == 'BRATS2024':
    #         csv_writer.writerow(['WT Dice','ET Dice','TC Dice','NETC Dice','SNFH Dice','RC Dice',
    #                          'WT HD95','ET HD95','TC HD95','NETC HD95','SNFH HD95','RC HD95'])
    #     else:
    #         csv_writer.writerow(['WT Dice','TC Dice','ET Dice','ETPro Dice',
    #                          'WT HD95','TC HD95','ET HD95','ETPro HD95'])
    #     file.close()
    #     for i, mask in enumerate(masks_test[::-1]):
    #         logging.info('{}'.format(mask_name[::-1][i]))
    #         file = open(csv_name, "a+")
    #         csv_writer = csv.writer(file)
    #         csv_writer.writerow([mask_name[::-1][i]])
    #         file.close()
    #         dice_score, hd95_score = test_dice_hd95_softmax(
    #             test_loader,
    #             model,
    #             dataname=args.dataname,
    #             feature_mask=mask,
    #             mask_name=mask_name[::-1][i],
    #             csv_name=csv_name,
    #         )
    #         test_dice_score.update(dice_score)
    #         test_hd95_score.update(hd95_score)

    #     logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
    #     logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))

    # ##########Test the best epoch model
    # file_name = os.path.join(ckpts, 'model_best.pth')
    # checkpoint = torch.load(file_name)
    # logging.info('best epoch: {}'.format(checkpoint['epoch']+1))
    # model.load_state_dict(checkpoint['state_dict'])
    # test_best_score = AverageMeter()
    # with torch.no_grad():
    #     logging.info('###########test validate best model###########')
    #     for i, mask in enumerate(masks_test[::-1]):
    #         logging.info('{}'.format(mask_name[::-1][i]))
    #         dice_best_score = test_softmax(
    #                         test_loader,
    #                         model,
    #                         dataname = args.dataname,
    #                         feature_mask = mask,
    #                         mask_name = mask_name[::-1][i])
    #         test_best_score.update(dice_best_score)
    #     logging.info('Avg scores: {}'.format(test_best_score.avg))
    if local_rank == 0:
        logging.info('########### Testing the LAST model ###########')
        file_name = os.path.join(ckpts, 'model_last.pth')

        if not os.path.exists(file_name):

            logging.error("model_best.pth not found! Testing model_last.pth instead.")
            file_name = os.path.join(ckpts, 'model_last.pth')
            if not os.path.exists(file_name):
                logging.error("model_last.pth not found either! Exiting test.")
                return

        checkpoint = torch.load(file_name, map_location=device, weights_only=False)

        logging.info('Loading best model from epoch: {}'.format(checkpoint['epoch']))
        model.module.load_state_dict(checkpoint['state_dict'])
        test_dice_score = AverageMeter()
        test_hd95_score = AverageMeter()
        csv_name = os.path.join(csvpath, '{}.csv'.format(args.model))
        with torch.no_grad():

            logging.info('###########test last epoch model###########')
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            # csv_writer.writerow(
            #     ['WT Dice', 'TC Dice', 'ET Dice', 'ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95', 'ETPro HD95'])
            if args.dataname == 'BRATS2024':
                csv_writer.writerow(['WT Dice', 'ET Dice', 'TC Dice', 'RC Dice',
                                     'WT HD95', 'ET HD95', 'TC HD95', 'RC HD95'])
            else:
                csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice', 'ETPro Dice',
                                     'WT HD95', 'TC HD95', 'ET HD95', 'ETPro HD95'])
            file.close()
            for i, mask in enumerate(masks_test[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                csv_writer.writerow([mask_name[::-1][i]])
                file.close()
                dice_score, hd95_score = test_dice_hd95_softmax(
                    test_loader,
                    model,
                    dataname=args.dataname,
                    feature_mask=mask,
                    mask_name=mask_name[::-1][i],
                    csv_name=csv_name

                )
                test_dice_score.update(dice_score)
                test_hd95_score.update(hd95_score)

            logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
            logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))


if __name__ == '__main__':
    main()
