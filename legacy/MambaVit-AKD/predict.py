import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import nibabel as nib
import scipy.misc
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch  # 用于创建图例
import matplotlib.cm as cm
import cv2
import csv
from medpy.metric import hd95
cudnn.benchmark = True

path = os.path.dirname(__file__)
from utils.generate import generate_snapshot


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ####post processing:
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4
    t5 = t4
    intersect5 = torch.sum(2 * (o5 * t5), dim=(1,2,3)) + eps
    denominator5 = torch.sum(o5, dim=(1,2,3)) + torch.sum(t5, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def test_softmax0(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        mask_name=None):
    print('coming!')
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'


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
        h_cnt = np.int64(np.ceil((H - 96) / (96 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(96 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 96)

        w_cnt = np.int64(np.ceil((W - 96) / (96 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(96 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 96)

        z_cnt = np.int64(np.ceil((Z - 96) / (96 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(96 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 96)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+96, w:w+96, z:z+96] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+96, w:w+96, z:z+96]
                    feature,logits,pred_part = model(x_input, mask)
                    pred[:, :, h:h+96, w:w+96, z:z+96] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
            # print(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print (msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg

def test_softmax1(
        test_loader,
        model,
        dataname = 'BRATS2020',
        feature_mask=None,
        mask_name=None):
    print('coming!')
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'


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
        h_cnt = np.int64(np.ceil((H - 96) / (96 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(96 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 96)

        w_cnt = np.int64(np.ceil((W - 96) / (96 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(96 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 96)

        z_cnt = np.int64(np.ceil((Z - 96) / (96 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(96 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 96)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+96, w:w+96, z:z+96] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training=False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+96, w:w+96, z:z+96]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h + 96, w:w + 96, z:z + 96] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)
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
    model.train()
    return vals_evaluation.avg

def test_softmax(
        i,
        test_loader,
        model,
        dataname='BRATS2020',
        feature_mask=None,
        mask_name=None):
    num=i
    print('coming!')
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

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

        h_cnt = np.int64(np.ceil((H - 96) / (96 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(96 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 96)

        w_cnt = np.int64(np.ceil((W - 96) / (96 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(96 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 96)

        z_cnt = np.int64(np.ceil((Z - 96) / (96 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(96 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 96)

        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + 96, w:w + 96, z:z + 96] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + 96, w:w + 96, z:z + 96]
                    feature, logits, pred_part = model(x_input, mask)
                    pred[:, :, h:h + 96, w:w + 96, z:z + 96] += pred_part
        pred = pred / weight
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

        for k, name in enumerate(names):
            # Save the segmentation result to nii.gz file
            pred_np = pred[k].cpu().numpy().astype(np.uint8)  # Convert tensor to numpy
            save_test_label(num,name, pred_np,target[k].cpu())  # Save to nii.gz

            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)
            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            logging.info(msg)
            print(msg)

    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg
# def save_nii(pred, save_filename, savepath, affine=None):
#     # Define the full file path with the save path
#     full_save_path = f"{savepath}/{save_filename}"
#
#     # Save the file as a NIfTI image
#     nifti_img = nib.Nifti1Image(pred, affine if affine is not None else np.eye(
#         4))  # Use identity matrix as default affine if not provided
#     nib.save(nifti_img, full_save_path)
#     print(f"Saved {full_save_path}")

def save_nii(pred, save_filename, savepath, reference_nii_path=None):
    # Define the full file path with the save path
    full_save_path = f"{savepath}/{save_filename}"

    # If a reference NIfTI file path is provided, load the NIfTI image and get affine matrix
    if reference_nii_path is not None:
        reference_nii = nib.load(reference_nii_path)
        affine = reference_nii.affine
        reference_shape = reference_nii.shape  # Get shape from reference
    else:
        affine = np.eye(4)  # Default to identity matrix if no reference is provided
        reference_shape = pred.shape  # Use the shape of the prediction image

    # Save the file as a NIfTI image
    nifti_img = nib.Nifti1Image(pred, affine)
    nib.save(nifti_img, full_save_path)
    print(f"Saved {full_save_path}")


def test_softmax001(
        i,
        test_loader,
        model,
        savepath,
        reference_nii_path,
        dataname='BRATS2020',
        feature_mask=None,
        mask_name=None):
    """Generate class probability heatmaps from model predictions with paper-style visualization"""
    num = i
    print('Generating class probability heatmaps with paper-style visualization...')

    model.eval()
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4  # background + WT + TC + ET
    elif dataname == 'BRATS2015':
        num_cls = 5

    # 创建保存目录
    os.makedirs(savepath, exist_ok=True)

    for i, data in enumerate(test_loader):
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, Z = x.size()

        # Calculate patch indices
        h_cnt = np.int64(np.ceil((H - 96) / (96 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(96 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 96)

        w_cnt = np.int64(np.ceil((W - 96) / (96 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(96 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 96)

        z_cnt = np.int64(np.ceil((Z - 96) / (96 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(96 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 96)

        # Create weight map
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + 96, w:w + 96, z:z + 96] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        # Get model predictions
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + 96, w:w + 96, z:z + 96]
                    feature, logits, pred_part = model(x_input, mask)
                    pred[:, :, h:h + 96, w:w + 96, z:z + 96] += pred_part

        # Normalize predictions by weight
        pred = pred / (weight + 1e-8)

        # Convert to numpy and apply softmax if needed
        pred_np = pred.detach().cpu().numpy()
        pred_np = np.exp(pred_np) / np.sum(np.exp(pred_np), axis=1, keepdims=True)  # Softmax

        # Process each sample in the batch
        for sample_idx in range(len(names)):
            slice_idx =89
            if slice_idx >= H:
                continue

            mri_slice = x[sample_idx, 0, slice_idx, :, :].cpu().numpy()
            pred_slice = pred_np[sample_idx, :, slice_idx, :, :]
            heatmap = create_paper_style_heatmap(mri_slice, pred_slice)

            save_medical_heatmap(
                num=num,
                heatmap=heatmap,
                name=names[sample_idx],
                save_dir=savepath,
                slice_idx=slice_idx,
                mask_name=mask_name  # 传递mask_name参数
            )





def create_paper_style_heatmap(mri_slice, pred_slice, alpha=0.6):
    # Normalize MRI
    mri_normalized = (mri_slice - np.percentile(mri_slice, 1)) / (
            np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1) + 1e-8)
    mri_rgb = np.stack([mri_normalized] * 3, axis=-1)

    # 使用Matplotlib的Rainbow/Jet colormap生成热力图
    cmap = plt.get_cmap('jet')  # 或者 'rainbow', 'viridis'

    # 合并多类别预测（ET优先级最高）
    combined_prob = np.zeros_like(pred_slice[0])
    if pred_slice.shape[0] > 3:  # 如果存在ET预测
        combined_prob = np.maximum(combined_prob, pred_slice[1] * 3)  # ET权重最高
    combined_prob = np.maximum(combined_prob, pred_slice[3] * 2)  # TC次之
    combined_prob = np.maximum(combined_prob, pred_slice[2] *1.9 )  # WT最低

    # 归一化并应用colormap
    combined_prob = (combined_prob - combined_prob.min()) / (combined_prob.max() - combined_prob.min() + 1e-8)
    heatmap = cmap(combined_prob)[:, :, :3]  # 取RGB，忽略alpha通道

    # 与MRI叠加
    blended = mri_rgb * (1 - alpha) + heatmap * alpha
    return np.clip(blended, 0, 1)

def save_medical_heatmap(num,heatmap, name, save_dir, slice_idx, mask_name=None):
    """
    保存热力图本身（无文字/图例/边框），文件名加入mask_name
    """
    plt.figure(figsize=(10, 10), dpi=300, frameon=False)
    plt.imshow(heatmap)
    plt.axis('off')

    # 构建文件名
    base_name = f"{name}_slice{slice_idx + 1}"
    if mask_name is not None:
        base_name = f"{base_name}_{mask_name[num]}"
    save_path = os.path.join(save_dir, f"{base_name}_clean.png")

    plt.savefig(save_path,
               bbox_inches='tight',
               pad_inches=0,
               dpi=300)
    plt.close()
    print(f"Heatmap saved to: {save_path}")

def test_softmax002(
        i,
        test_loader,
        model,
        savepath,  # Added savepath argument
        reference_nii_path,
        dataname='BRATS2020',
        feature_mask=None,
        mask_name=None):
    num = i
    print('coming!')
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

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

        h_cnt = np.int64(np.ceil((H - 96) / (96 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(96 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 96)

        w_cnt = np.int64(np.ceil((W - 96) / (96 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(96 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 96)

        z_cnt = np.int64(np.ceil((Z - 96) / (96 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(96 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 96)

        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + 96, w:w + 96, z:z + 96] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + 96, w:w + 96, z:z + 96]
                    feature, logits, pred_part = model(x_input, mask)
                    pred[:, :, h:h + 96, w:w + 96, z:z + 96] += pred_part
        pred = pred / weight

        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

        for k, name in enumerate(names):
            # Save the segmentation result to nii.gz file
            pred_np = pred[k].cpu().numpy().astype(np.uint8)
            save_filename = f"{name}_mask_{mask_name[num]}_seg.nii.gz"
            save_nii(pred_np, save_filename, savepath, reference_nii_path)

            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)
            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            logging.info(msg)
            print(msg)

    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg


def test_softmax2(
        i,
        test_loader,
        model,
        dataname='BRATS2020',
        feature_mask=None,
        mask_name=None):
    num=i
    print('coming!')
    H, W, T = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

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

        h_cnt = np.int64(np.ceil((H - 96) / (96 * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int64(96 * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - 96)

        w_cnt = np.int64(np.ceil((W - 96) / (96 * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int64(96 * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - 96)

        z_cnt = np.int64(np.ceil((Z - 96) / (96 * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int64(96 * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - 96)

        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + 96, w:w + 96, z:z + 96] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training = False
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + 96, w:w + 96, z:z + 96]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h + 96, w:w + 96, z:z + 96] += pred_part
        pred = pred / weight
        pred = pred[:, :, :H, :W, :T]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

        for k, name in enumerate(names):
            # Save the segmentation result to nii.gz file
            pred_np = pred[k].cpu().numpy().astype(np.uint8)  # Convert tensor to numpy
            save_test_label(num,name, pred_np,target[k].cpu())  # Save to nii.gz

            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)
            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            logging.info(msg)
            print(msg)

    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg


def crop_image_to_pred_size(ref_img, predict):
    # 获取预测图像的尺寸
    pred_shape = predict.shape  # 预测图像的形状 (depth, height, width)

    # 获取参考图像的尺寸
    ref_size = ref_img.GetSize()  # SimpleITK 图像的形状 (width, height, depth)

    # 计算裁剪的起始索引和结束索引
    start_idx = [(ref_size[i] - pred_shape[i]) // 2 if ref_size[i] > pred_shape[i] else 0 for i in
                 range(3)]  # 确保起始点不会为负数
    end_idx = [start_idx[i] + pred_shape[i] if start_idx[i] + pred_shape[i] <= ref_size[i] else ref_size[i] for i in
               range(3)]  # 确保结束点不超出参考图像大小

    # 裁剪参考图像，确保大小匹配
    crop_filter = sitk.RegionOfInterestImageFilter()
    crop_filter.SetSize([end_idx[0] - start_idx[0], end_idx[1] - start_idx[1], end_idx[2] - start_idx[2]])
    crop_filter.SetIndex(start_idx)
    cropped_ref_img = crop_filter.Execute(ref_img)

    return cropped_ref_img


# def save_test_label(num,patient_id, predict):
#     # 使用固定的 NIfTI 文件作为模板
#     fixed_nii_path = './BraTS2021_00000_flair.nii.gz'  # 固定的 .nii.gz 文件路径
#     ref_img = sitk.ReadImage(fixed_nii_path)  # 读取固定的模板文件
#
#     # 裁剪参考图像
#     ref_img_cropped = crop_image_to_pred_size(ref_img, predict)  # 裁剪参考图像
#
#     # 裁剪后转换维度顺序，从 (width, height, depth) 转换为 (depth, height, width)
#     ref_img_array = sitk.GetArrayFromImage(ref_img_cropped)  # 转换为 NumPy 数组 (depth, height, width)
#     ref_img_array = np.transpose(ref_img_array, (2, 1, 0))  # 转换为 (width, height, depth)
#
#     # 将转换后的 NumPy 数组转换回 SimpleITK 图像
#     ref_img_transposed = sitk.GetImageFromArray(ref_img_array)
#
#     # 将预测结果转换为 SimpleITK 图像
#     label_nii = sitk.GetImageFromArray(predict)
#
#     # 复制裁剪后参考图像的信息
#     label_nii.CopyInformation(ref_img_transposed)
#
#     # 创建 num 文件夹路径
#     folder_path = os.path.join('./label_folder', str(num))
#     # 如果文件夹不存在则创建
#     os.makedirs(folder_path, exist_ok=True)
#     # 保存为 NIfTI 文件到指定文件夹
#     sitk.WriteImage(label_nii, os.path.join(folder_path, f"{patient_id}.nii.gz"))
def save_test_label(num, patient_id, predict, mask):
    # 使用固定的 NIfTI 文件作为模板
    fixed_nii_path = './BraTS2021_00000_flair.nii.gz'  # 固定的 .nii.gz 文件路径
    ref_img = sitk.ReadImage(fixed_nii_path)  # 读取固定的模板文件

    # 裁剪参考图像
    ref_img_cropped = crop_image_to_pred_size(ref_img, predict)  # 裁剪参考图像

    # 裁剪后转换维度顺序，从 (width, height, depth) 转换为 (depth, height, width)
    ref_img_array = sitk.GetArrayFromImage(ref_img_cropped)  # 转换为 NumPy 数组 (depth, height, width)
    ref_img_array = np.transpose(ref_img_array, (2, 1, 0))  # 转换为 (width, height, depth)

    # 将转换后的 NumPy 数组转换回 SimpleITK 图像
    ref_img_transposed = sitk.GetImageFromArray(ref_img_array)

    # 将预测结果转换为 SimpleITK 图像
    label_nii = sitk.GetImageFromArray(predict)

    # 复制裁剪后参考图像的信息
    label_nii.CopyInformation(ref_img_transposed)

    # 保存预测结果为 NIfTI 文件
    folder_path = os.path.join('./label_folder_cnn', str(num))
    os.makedirs(folder_path, exist_ok=True)
    sitk.WriteImage(label_nii, os.path.join(folder_path, f"{patient_id}_prediction.nii.gz"))

    # 将 mask 直接转换为 SimpleITK 图像并保存
    mask_nii = sitk.GetImageFromArray(mask)  # 直接传递 NumPy 数组，无需 .cpu().numpy()
    mask_nii.CopyInformation(ref_img_transposed)  # 复制信息
    sitk.WriteImage(mask_nii, os.path.join(folder_path, f"{patient_id}_mask.nii.gz"))  # 保存为 .nii.gz 文件

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
def test_dice_hd95_softmax(
        test_loader,
        model,
        dataname='BRATS2020',
        feature_mask=None,
        mask_name=None,
        csv_name=None,
):
    H, W, T = 240, 240, 155
    model.eval()
    vals_dice_evaluation = AverageMeter()
    vals_hd95_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    patch_size=96
    one_tensor = torch.ones(1, 96, 96, 96).float().cuda()

    if dataname in ['BRATS2020']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    # elif dataname == '/home/sjj/MMMSeg/BraTS/BRATS2015':
    #     num_cls = 5
    #     class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    #     class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

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
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda()
        model.is_training = False


        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size]
                    feature, logits, pred_part = model(x_input, mask)
                    pred[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :Z]
        pred = torch.argmax(pred, dim=1)

        if dataname in ['BRATS2020']:
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
            print(msg)
            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_hd95_evaluation.avg)])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    logging.info(msg)
    model.train()
    return vals_dice_evaluation.avg, vals_hd95_evaluation.avg