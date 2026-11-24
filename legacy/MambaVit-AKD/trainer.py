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

import model_RFNet
import model_mamba_modalitys

from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax0
from loss import  softmax_kl_loss, prototype_loss, attention_loss

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='./data/BraTS2020_Data', type=str)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--savepath', default='./savelog_2020_modelBaseline_mamba_ST', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=219, type=int)
parser.add_argument('--region_fusion_start_epoch', default=200, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--student_channels', default=4, type=int)
parser.add_argument('--teacher_channels', default=4, type=int)
parser.add_argument('--n_filters', default=16, type=int)
parser.add_argument('--teachermodel_path', default='./savelog_2021_RFNet_teacher/best_model.pth', type=str)
parser.add_argument('--T', default=10, type=int)
parser.add_argument('--num_classes', default=4, type=int)
parser.add_argument('--proto_weight', type=float, default=0.1, help='proto loss weight')
parser.add_argument('--kd_weight', type=float, default=10, help='kd loss weight')
parser.add_argument('--attn_weight', type=float, default=0.1, help='attention loss weight')
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((96,96,96)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
args.val_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [True, True, False,False],[True, True, False, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = [  't2', 't1c', 't1', 'flair',
               'flairt1ce',
            'flairt1cet2',
            'flairt1cet1t2']
# print (masks_torch.int())
def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    num_cls = 4
    model = model_baseline.Model(
            num_cls=num_cls,
            in_channels=4,
            img_size=(96,96,96),
            hidden_size=768,
            mlp_dim=3072,
            num_heads= 12,
            pos_embed= "perceptron",
            dropout_rate= 0.0,)
    # print (model)
    model.train()
    model.cuda()
    teacher_model=model_RFNet.Model(
            num_cls=num_cls,
            )
    teacher_model.cuda()
    # teacher_model.load_state_dict(torch.load(args.teachermodel_path))
    # 加载 checkpoint 文件
    checkpoint = torch.load(args.teachermodel_path)
    # 检查文件是否包含 'state_dict'
    if 'state_dict' in checkpoint:
        # 提取模型权重
        state_dict = checkpoint['state_dict']
    else:
        # 直接使用整个文件作为 state_dict
        state_dict = checkpoint
    # 加载模型权重
    # 修改 state_dict 层名，去掉 'module.' 前缀
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # 加载修改后的 state_dict
    teacher_model.load_state_dict(state_dict)
    # 切换模型到评估模式
    teacher_model.eval()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    ##########Setting data
    train_file = 'train.txt'
    test_file = 'test.txt'
    val_file='valid.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                  train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.val_transforms, root=args.datapath, val_file=val_file)
    print("train_set include", len(train_set), "datas!")
    print("val_set include", len(val_set), "datas!")
    print("test_set include", len(test_set), "datas!")
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False, )
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)


    ##########Training
    if args.resume is None:
        start = time.time()
        torch.set_grad_enabled(True)
        # # # 加载整个 checkpoint 文件
        # checkpoint = torch.load('./savelog_2021_modelBaseline_ST/model_last.pth')
        # # # 提取模型权重
        # model_weights = checkpoint['state_dict']
        # # # 加载模型权重到模型
        # model.load_state_dict(model_weights)
        # # # 获取保存的 epoch 数，继续训练时使用
        # start_epoch = checkpoint['epoch']
        start_epoch=0
        logging.info('#############training############')
        iter_per_epoch = args.iter_per_epoch
        train_iter = iter(train_loader)
        best_dice_score=0.0
        for epoch in range(start_epoch,args.num_epochs):
            step_lr = lr_schedule(optimizer, epoch)
            writer.add_scalar('lr', step_lr, global_step=(epoch+1))
            b = time.time()
            for i in range(iter_per_epoch):
                step = (i+1) + epoch*iter_per_epoch
                ###Data load
                try:
                    data = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    data = next(train_iter)
                x, target, mask, mask_all = data[:4]
                # print(x.shape)
                # print(target.shape)
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                mask_all = mask_all.cuda(non_blocking=True)

                model.is_training = True
                feature,logits,fuse_pred, sep_preds, prm_preds = model(x, mask)
                teacher_model.is_training = False
                with torch.no_grad():
                    feature_t, logits_t, fuse_pred_t = teacher_model(x, mask_all)
                # kd loss 实现了一个计算KL散度的函数，用于衡量两个分布（input_logits 和 target_logits）之间的差异
                kd_loss = softmax_kl_loss(logits / args.T, logits_t / args.T).mean()
                # proto loss 用于比较教师模型和学生模型的特征相似性，并将它们的原型相似性图进行对齐
                sim_map_s, sim_map_t, proto_loss = prototype_loss(feature, feature_t, target, args.num_classes)
                # attention loss
                attn_loss = attention_loss(feature.cuda(), feature_t.cuda())
                # apkd_loss
                apkd_loss = args.kd_weight * kd_loss + args.proto_weight * proto_loss+args.attn_weight * attn_loss
                ###Loss compute
                fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)   #加权的交叉熵损失
                fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)  #dice_loss
                fuse_loss = fuse_cross_loss + fuse_dice_loss

                sep_cross_loss = torch.zeros(1).cuda().float()
                sep_dice_loss = torch.zeros(1).cuda().float()
                for sep_pred in sep_preds:
                    sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                    sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
                sep_loss = sep_cross_loss + sep_dice_loss

                prm_cross_loss = torch.zeros(1).cuda().float()
                prm_dice_loss = torch.zeros(1).cuda().float()
                for prm_pred in prm_preds:
                    prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                    prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
                prm_loss = prm_cross_loss + prm_dice_loss

                if epoch < args.region_fusion_start_epoch:
                    loss = fuse_loss * 0.0 + sep_loss + prm_loss+apkd_loss
                else:
                    loss = fuse_loss + sep_loss + prm_loss+apkd_loss

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
                writer.add_scalar('kd_loss', kd_loss.item(), global_step=step)
                writer.add_scalar('proto_loss', proto_loss.item(), global_step=step)

                msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch), args.num_epochs, (i+1), iter_per_epoch, loss.item())
                msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
                msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
                msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
                msg += 'kd_loss:{:.4f}, proto_loss:{:.4f},'.format(kd_loss.item(), proto_loss.item())
                logging.info(msg)
            logging.info('train time per epoch: {}'.format(time.time() - b))
            # 保存最后的模型为 model_last.pth
            file_name = os.path.join(ckpts, 'model_last.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, file_name)
            ##########model save
            if epoch >= 180:
                val_score = AverageMeter()
                # 初始化一个变量来保存最佳 dice_score
                with torch.no_grad():
                    logging.info('###########val set wi/wo postprocess###########')
                    for i, mask in enumerate(masks):
                        logging.info('{}'.format(mask_name[i]))
                        dice_score = test_softmax0(
                            val_loader,
                            model,
                            dataname=args.dataname,
                            feature_mask=mask
                        )
                        val_score.update(dice_score)

                    avg_dice_score = val_score.avg  # 计算平均分数  假设avg_dice_score = [0.85, 0.76, 0.68, 0.81]
                    avg_dice_score = sum(avg_dice_score) / len(avg_dice_score)
                    logging.info('Avg scores: {}'.format(avg_dice_score))

                    # 如果当前的 avg_dice_score 比之前的 best_dice_score 高，保存为 best_model.pth
                    if avg_dice_score > best_dice_score:
                        best_dice_score = avg_dice_score  # 更新最佳分数
                        best_model_path = os.path.join(ckpts, 'best_model.pth')
                        torch.save({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optim_dict': optimizer.state_dict(),
                        }, best_model_path)
                        logging.info(f'Saving new best model with dice score: {best_dice_score}')
                    best_dice_score=avg_dice_score


        msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
        logging.info(msg)

    ##########test#######
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        test_score = AverageMeter()
        with torch.no_grad():
            logging.info('###########test set wi post process###########')
            for i, mask in enumerate(masks[::-1]):
                logging.info('{}'.format(mask_name[::-1][i]))
                dice_score = test_softmax0(
                    test_loader,
                    model,
                    dataname=args.dataname,
                    feature_mask=mask,
                    mask_name=mask_name[::-1][i])
                test_score.update(dice_score)
            logging.info('Avg scores: {}'.format(test_score.avg))

if __name__ == '__main__':
    main()
