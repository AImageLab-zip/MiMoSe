import logging
import os
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from medpy.metric import hd95
import csv
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk

# from utils.visualize import visualize_heads
# from visualizer import get_local

# get_local.activate()

cudnn.benchmark = True

path = os.path.dirname(__file__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
patch_size = 80


def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)
    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 1.0
            # follow ACN and SMU-Net
            # return 373.12866
            # follow nnUNet
    elif num_pred == 0 and num_ref != 0:
        return 1.0
        # follow ACN and SMU-Net
        # return 373.12866
        # follow in nnUNet
    else:
        return hd95(pred, ref, (1, 1, 1))


def cal_hd95(output, target):
    # whole tumor
    mask_gt = (target != 0).astype(int)
    mask_pred = (output != 0).astype(int)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = ((target == 1) | (target == 3)).astype(int)
    mask_pred = ((output == 1) | (output == 3)).astype(int)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (target == 3).astype(int)
    mask_pred = (output == 3).astype(int)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    mask_gt = (target == 3).astype(int)
    if np.sum((output == 3).astype(int)) < 500:
        mask_pred = (output == 3).astype(int) * 0
    else:
        mask_pred = (output == 3).astype(int)
    hd95_enhpro = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return (hd95_whole, hd95_core, hd95_enh, hd95_enhpro)


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
        o4 = o3 * 0.0
    else:
        o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1),
                               torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def test_softmax(
        test_loader,
        model,
        dataname='/home/sjj/M2FTrans/BraTS/BRATS2020',
        feature_mask=None,
        mask_name=None):
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

    if dataname in ['/home/sjj/MMMSeg/BraTS/BRATS2021', '/home/sjj/M2FTrans/BraTS/BRATS2020',
                    '/home/sjj/M2FTrans/BraTS/BRATS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'

    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size)

        w_cnt = np.int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size)

        z_cnt = np.int(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.module.is_training = False

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :Z]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


EPS = 1e-8


def _to_numpy_bool(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x.astype(np.uint8)


def dice_binary_torch(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    p = pred_mask.reshape(-1).float()
    g = gt_mask.reshape(-1).float()
    if g.sum() < 0.5:
        return None
    inter = 2.0 * (p * g).sum()
    denom = p.sum() + g.sum() + EPS
    return (inter / denom).item()


def hd95_binary_numpy_brats(pred_mask_np, gt_mask_np, spacing=(1, 1, 1)):
    pred = np.asarray(pred_mask_np)
    gt = np.asarray(gt_mask_np)

    if pred.ndim == 4 and pred.shape[0] == 1: pred = pred[0]
    if gt.ndim == 4 and gt.shape[0] == 1: gt = gt[0]

    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f'hd95 expects 3D [H,W,D], got pred.ndim={pred.ndim}, gt.ndim={gt.ndim}')

    pred = pred.astype(np.bool_)
    gt = gt.astype(np.bool_)

    num_pred = int(pred.sum())
    num_gt = int(gt.sum())

    if num_gt == 0:
        if num_pred == 0:
            return 0.0
        else:
            return 1.0
    else:
        if num_pred == 0:
            return 1.0
        else:
            return float(hd95(pred, gt, voxelspacing=spacing))


def aggregate_avg(per_case_rows):
    import numpy as np
    arr = []
    for row in per_case_rows:
        new_row = []
        for v in row:
            if (v is None) or (isinstance(v, float) and np.isnan(v)):
                new_row.append(np.nan)
            else:
                new_row.append(v)
        arr.append(new_row)
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr, axis=0)


def test_dice_hd95_softmax(
        test_loader,
        model,
        dataname='/home/z/文档/test/dyq/M2FTrans-main/BraTS/BRATS2020_Training_none_npy',
        feature_mask=None,
        mask_name=None,
        csv_name=None,
):
    H, W, T = 240, 240, 155
    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    inference_time_meter = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().to(device)

    if dataname in ['BRATS2018', 'BRATS2020', 'BRATS2021']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    # elif dataname == '/home/sjj/MMMSeg/BraTS/BRATS2015':
    #     num_cls = 5
    #     class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    #     class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            target = data[1].to(device)
            x = data[0].to(device)
            names = data[-1]
            if feature_mask is not None:
                mask = torch.from_numpy(np.array(feature_mask))
                mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
            else:
                mask = data[2]
            mask = mask.to(device)
            _, _, H, W, Z = x.size()
            #########get h_ind, w_ind, z_ind for sliding windows
            h_cnt = np.int_(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
            h_idx_list = range(0, h_cnt)
            h_idx_list = [h_idx * np.int_(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
            h_idx_list.append(H - patch_size)

            w_cnt = np.int_(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
            w_idx_list = range(0, w_cnt)
            w_idx_list = [w_idx * np.int_(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
            w_idx_list.append(W - patch_size)

            z_cnt = np.int_(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
            z_idx_list = range(0, z_cnt)
            z_idx_list = [z_idx * np.int_(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
            z_idx_list.append(Z - patch_size)

            #####compute calculation times for each pixel in sliding windows
            weight1 = torch.zeros(1, 1, H, W, Z).float().to(device)
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += one_tensor
            weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            #####evaluation
            pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device)
            # model.module.is_training=False
            model.is_training = False

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size]
                        pred_part = model(x_input, mask)
                        pred[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += pred_part
            pred = pred / weight
            b = time.time()
            pred = pred[:, :, :H, :W, :Z]
            pred = torch.argmax(pred, dim=1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            elapsed_time = end_time - start_time
            inference_time_meter.update(elapsed_time)

            if dataname in ['BRATS2018', 'BRATS2020', 'BRATS2021']:
                scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
                scores_hd95 = np.array(cal_hd95(pred[0].cpu().numpy(), target[0].cpu().numpy()))

            for k, name in enumerate(names):
                msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
                msg += '{:>20}, '.format(name)

                vals_separate.update(scores_separate[k])
                vals_dice_evaluation.update(scores_evaluation[k])
                vals_hd95_evaluation.update(scores_hd95)
                msg += 'DSC: '
                msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
                msg += ', HD95: '
                msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_hd95)])
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                csv_writer.writerow(
                    [scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2], scores_evaluation[k][3], \
                     scores_hd95[0], scores_hd95[1], scores_hd95[2], scores_hd95[3]])
                file.close()
                # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

                logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])

    time_msg = '\n[Efficiency] Average Inference Time per Image: {:.4f} s | FPS: {:.2f}'.format(
        inference_time_meter.avg, 1.0 / inference_time_meter.avg)
    print(msg + time_msg)
    logging.info(msg + time_msg)
    # print (msg)
    # logging.info(msg)
    model.train()
    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg

# def test_softmax_visualize(
#         test_loader,
#         model,
#         dataname = '/home/sjj/M2FTrans/BraTS/BRATS2020',
#         feature_mask=None,
#         mask_name=None,
#         writer = None,
#         step = 0):

#     H, W, T = 240, 240, 155
#     model.eval()
#     vals_evaluation = AverageMeter()
#     vals_separate = AverageMeter()
#     one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().cuda()

#     if dataname in ['/home/sjj/M2FTrans/BraTS/BRATS2020', '/home/sjj/M2FTrans/BraTS/BRATS2018']:
#         num_cls = 4
#         class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
#         class_separate = 'ncr_net', 'edema', 'enhancing'

#     global_step = step


#     for i, data in enumerate(test_loader):
#         target = data[1].cuda()
#         x = data[0].cuda()
#         names = data[-1]
#         if feature_mask is not None:
#             mask = torch.from_numpy(np.array(feature_mask))
#             mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
#         else:
#             mask = data[2]
#         mask = mask.cuda()
#         _, _, H, W, Z = x.size()
#         #########get h_ind, w_ind, z_ind for sliding windows
#         h_cnt = np.int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
#         h_idx_list = range(0, h_cnt)
#         h_idx_list = [h_idx * np.int(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
#         h_idx_list.append(H - patch_size)

#         w_cnt = np.int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
#         w_idx_list = range(0, w_cnt)
#         w_idx_list = [w_idx * np.int(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
#         w_idx_list.append(W - patch_size)

#         z_cnt = np.int(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
#         z_idx_list = range(0, z_cnt)
#         z_idx_list = [z_idx * np.int(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
#         z_idx_list.append(Z - patch_size)

#         #####compute calculation times for each pixel in sliding windows
#         weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
#         for h in h_idx_list:
#             for w in w_idx_list:
#                 for z in z_idx_list:
#                     weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
#         weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

#         #####evaluation
#         pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
#         model.module.is_training=False

#         for h in h_idx_list:
#             for w in w_idx_list:
#                 for z in z_idx_list:
#                     x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
#                     get_local.clear()
#                     pred_part = model(x_input, mask)
#                     cache = get_local.cache
#                     logging.info(list(cache.keys()))
#                     attention_maps = cache['MaskedAttention.forward']
#                     #attention_maps = cache['MultiHeadCrossAttention.forward']
#                     logging.info(len(attention_maps))
#                     for i in range(0, len(attention_maps)):
#                         visualize_heads(writer,attention_maps[i], cols=5, step=global_step, num=i)
#                         # logging.info(attention_maps[i].max())
#                         # logging.info(attention_maps[i].min())
#                     global_step = global_step + 1
#                     # logging.info(len(attention_map))
#                     # logging.info(attention_map[0].shape)
#                     # logging.info(h)
#                     # logging.info(w)
#                     # logging.info(z)
#                     # logging.info(patch_size)
#                     pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
#         pred = pred / weight
#         b = time.time()
#         pred = pred[:, :, :H, :W, :T]
#         pred = torch.argmax(pred, dim=1)

#         if dataname in ['/home/sjj/M2FTrans/BraTS/BRATS2020', '/home/sjj/M2FTrans/BraTS/BRATS2018']:
#             scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
#         for k, name in enumerate(names):
#             msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
#             msg += '{:>20}, '.format(name)

#             vals_separate.update(scores_separate[k])
#             vals_evaluation.update(scores_evaluation[k])
#             msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
#             #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

#             logging.info(msg)
#     msg = 'Average scores:'
#     msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
#     #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
#     print (msg)
#     logging.info(msg)
#     model.train()
#     return vals_evaluation.avg, global_step
