import argparse
import os
import pathlib
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from monai.data import decollate_batch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from dataset.brats import get_datasets23_train_rf_withvalidtest
from loss import EDiceLoss
from loss.dice import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, setup_seed, save_last
from model.Unet import Unet_missing
from torch.cuda.amp import autocast as autocast
import wandb
from dataset.transforms import *
from model.mask_utils import MaskEmbeeding1

val_check = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 260, 270, 275, 280, 285, 290, 295, 300]
parser = argparse.ArgumentParser(description='CNNNET BRATS 2023 Training')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--mdp', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--patch_shape', default=128, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')                   
parser.add_argument('--part', '--data_part', default=1, type=float, help='low shot, data proportion')
parser.add_argument('--all_data', default=False, action='store_true', help ='if train, when part != 1, do semi-supervise when finetune')
parser.add_argument('--semi_proportion', default=1, type=float, help='the proportion of the data used without label')
parser.add_argument('--use_weight', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--wd', '--weight-decay', default=0.00001, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--model_type', type=str, default='cnnnet', choices=['vtnet', 'cnnnet'])
parser.add_argument('--feature_level', default=2, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--exp_name', default='baseline3_real', type=str, help='exp name')
parser.add_argument('--val', default=20, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=999, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_costum.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=None, type=str, help='path to checkpoint')
parser.add_argument('--deep_supervised', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use_checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--weight_kl', default=1, type=float, help="train sub")
parser.add_argument('--debug', action='store_true', default=False)

class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8) 
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr

def KL_loss(pred1, pred2):
    # compute a weighted Kullback-Leibler (KL) divergence loss between two predictions
    sigmoid1 = torch.sigmoid(pred1)
    sigmoid2 = torch.sigmoid(pred2)
    
    kl_loss = F.kl_div(torch.log(sigmoid1), sigmoid2.detach(), reduction='none')
    
    # weight as the difference in entropy, clamping negative values to 0
    entropy1 = -1 * sigmoid1.detach() * torch.log(sigmoid1.detach())
    entropy2 = -1 * sigmoid2.detach() * torch.log(sigmoid2.detach())
    weight = entropy1 - entropy2
    weight[weight < 0] = 0

    kl_loss = kl_loss * weight
    
    return kl_loss
  
def mse_loss(fine_pred, missing_pred):  
    # calculate the Mean Squared Error (MSE) loss between two predictions for each channel
    loss1 = F.mse_loss(fine_pred[:, 0], missing_pred[:, 0], reduction='mean') 
    loss2 = F.mse_loss(fine_pred[:, 1], missing_pred[:, 1], reduction='mean') 
    loss3 = F.mse_loss(fine_pred[:, 2], missing_pred[:, 2], reduction='mean') 
    
    return loss1 + loss2 + loss3
    
def main(args):
    args.train_transforms = 'Compose([RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0),])'
    args.train_transforms = 'Compose([RandomIntensityChange((0.1,0.1)), RandomFlip(0),])'
    args.train_transforms = eval(args.train_transforms)

    ### Setup
    ########## setting seed
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    setup_seed(args.seed)

    ########## setup
    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    print(args)

    ##########init wandb
    if not args.debug:
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        wandb_name_and_id = f'BraTS23_m3ae_train_epoch{args.epochs}_model_type{args.model_type}_jobid{slurm_job_id}'
        wandb.init(
            project="SegmentationMM",
            name=wandb_name_and_id,
            #entity="maxillo",
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "M3AE",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "model_type": args.model_type,
                "epochs": args.epochs,
                "mdp": args.mdp,
            }
        )

    ### Model Preparation
    model_1 = Unet_missing(input_shape = [128,128,128], init_channels = 16, out_channels=3, mdp=3, pre_train = False, deep_supervised = args.deep_supervised, patch_shape = args.patch_shape)
    model_1 = nn.DataParallel(model_1)
    args.checkpoint = '/path/to/missing_modalities/m3ae/runs/m3ae_pretrain/model_1model_best_599.pth.tar'
    ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    del ck['state_dict']['module.unet.up1conv.weight'] #in pre-training out-channels = 4, during training out-channels = 3
    del ck['state_dict']['module.unet.up1conv.bias'] 
    model_1.load_state_dict(ck['state_dict'], strict=False)
    model_1 = model_1.module
    model_1 = nn.DataParallel(model_1)
    model_1 = model_1.cuda()
    #model_1.module.limage = model_1.module.limage.cpu()
    model_1.module.raw_input = model_1.module.raw_input.cpu()
    
    print(f"total number of trainable parameters {count_parameters(model_1)}")

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    ########## setting learning scheduler and optimizer
    ### Criterion
    criterion = EDiceLoss().cuda() #Dice Loss
    criterian_val = EDiceLoss_Val().cuda()
    CE_L = torch.nn.BCELoss(reduction = "none").cuda() #BCE loss
    metric = criterian_val.metric
    print(metric)
    
    ### Optimizer and Scheduler
    limage = []
    ori_para = []
    
    for pname, p in model_1.named_parameters():
        if pname.endswith("limage"):
            limage.append(p)
            #ori_para.append(p)
        else:
            ori_para.append(p)
    optimizer = torch.optim.Adam(ori_para, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    ########## setting data
    #if args.val != 0:
    full_train_dataset, l_val_dataset, l_test_dataset, _,_labeled_name_list = get_datasets23_train_rf_withvalidtest(args.seed, fold_number=args.fold, part = args.part, all_data = args.all_data, patch_shape = args.patch_shape)
        #print(_labeled_name_list)
    #else:
        #full_train_dataset, l_val_dataset, _labeled_name_list = get_datasets_train(args.seed, fold_number=args.fold, part = args.part)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=1, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(l_test_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    print("Test dataset number of batch:", len(test_loader))

    ########## training
    best_1 = 0.0
    patients_perf = []
    best_loss = 10000000
    start_epoch = 0
    limage = model_1.module.limage.detach().cpu().numpy()
    raw_input = model_1.module.raw_input.cpu().numpy()
    mean_results = torch.ones(15, 3)
    
    print("start training now!")
        
    #########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_1.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_1 = checkpoint['best_1']
        print(f"Reloaded from epoch {checkpoint['epoch']}!")
    
    
    for epoch in range(start_epoch, args.epochs):
        if epoch < start_epoch:
            if scheduler is not None:
                scheduler.step()
            continue
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')
            kl_loss = AverageMeter('Loss', ':.4e')
            epoch_loss = 0
            
            model_1.train()
            
            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]")
            end = time.perf_counter()

            for i, batch in enumerate(zip(train_loader)):
                # measure data loading time
                data_time.update(time.perf_counter() - end)
                #if not batch[0]["patient_id"][0] in _labeled_name_list and (np.random.random() > args.semi_proportion): # or epoch < 50):
                #    continue
                    
                inputs_S1, labels_S1 = batch[0]["image"].numpy(), batch[0]["label"].numpy() #image = (B, 4, 128, 128, 128), label = (B, 3, 128, 128, 128)
                patch_locations = batch[0]["crop_indexes"]
                
                inputs_S1 = inputs_S1.repeat(args.batch_size, axis = 0) # double in batch channel to have two missing modal instantiations for each sample
                mask_codes = [] #(1, 2) -> numbers between 0%14 to identify the missing modality scenario
                for ci in range(args.batch_size):
                    #two random missing modal instantiations for sample inputs_S1[ci]
                    mask = MaskEmbeeding1(1, raw_input = raw_input, mdp = args.mdp, mask = False, patch_shape = args.patch_shape)   # mask=False -> do not do patch mask but do modal mask
                    inputs_S1[ci] = inputs_S1[ci] * mask + limage[:,:,patch_locations[0][0]: patch_locations[0][1],patch_locations[1][0]: patch_locations[1][1],patch_locations[2][0]: patch_locations[2][1]] * (1-mask)# not 
                    mask_code = mask.mean((2,3,4)) #(1, 4) = 0/1 to identify modalities masked
                    for l in range(mask_code.shape[0]):
                        mask_codes.append(int(mask_code[l, 0] * 8 + mask_code[l,1]*4 + mask_code[l,2]*2 + mask_code[l,3]*1 -1 )) 
                
                inputs_S1, labels_S1 = args.train_transforms([inputs_S1.transpose(0,2,3,4,1), labels_S1.transpose(0,2,3,4,1)])
                inputs_S1, labels_S1 = [torch.from_numpy(np.ascontiguousarray(x.transpose(0,4,1,2,3))).float() for x in [inputs_S1, labels_S1]]
                labels_S1 = labels_S1.repeat(args.batch_size,1,1,1,1)
                patch_locations = [[s.repeat(args.batch_size) for s in l] for l in patch_locations] #repeat the patch locations across the batch dimension

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                optimizer.zero_grad()
                # segs_S1 = [out4, out3, uout] = last 3 results of the decoder = 3 x (B, 3, 128, 128, 128)
                # style = [c2d, c3d, c4d] = last 3 results of the encoder = 0: (B, 64, 32, 32, 32), 1: (B, 128, 16, 16, 16), 2: (B, 128, 16, 16, 16)
                # content = c4d = output encoder = (B, 128, 16, 16, 16)
                segs_S1, _, style, content = model_1(inputs_S1, patch_locations) 
                
                style.append(segs_S1[0]) # style = [c2d, c3d, c4d, out4]
                loss_ = 0
                
                # Dice loss
                if not args.deep_supervised: #False
                    loss_ += criterion(segs_S1, labels_S1)      
                else: #True => segs_S1 = [out4, out3, uout]
                    for l in segs_S1:
                        loss_ += criterion(l, labels_S1)

                for l in range(segs_S1[0].shape[0]): #
                    closs = criterion(segs_S1[0][l:l+1], labels_S1[l:l+1])
                    metric_ = metric(segs_S1[0][l:l+1], labels_S1[l:l+1])
                    mean_results[mask_codes[l], :] = mean_results[mask_codes[l], :] * 0.99 + torch.tensor([l.detach().cpu() for l in metric_[0]]) * 0.01
                
                if args.batch_size == 2:
                    if args.use_weight and epoch > 50: #False
                        weight1 = []
                        weight2 = []
                        for ii in range(labels_S1.shape[1]):
                            weight1.append(CE_L(torch.sigmoid(segs_S1[-1][:1, ii, :, :, :].detach()), labels_S1[:1, ii, :, :, :]))
                            weight2.append(CE_L(torch.sigmoid(segs_S1[-1][1:, ii, :, :, :].detach()), labels_S1[1:, ii, :, :, :]))
                        weight1 = 1-torch.cat(weight1, dim = 0)
                        weight2 = 1-torch.cat(weight2, dim = 0)
                        weight1[weight1<0] = 0
                        weight2[weight2<0] = 0
                        weight = (weight1 + weight2) / 2
                        weight = torch.mean(weight, 0, keepdim = True)
                        weight = weight.unsqueeze(0)
                        weight = F.interpolate(weight,size=(16,16,16), mode='trilinear', align_corners=True)[0]
                        #F.Upsample(style[args.feature_level][0], scale_factor=8, mode='trilinear', align_corners=True)
                        loss_kl = (F.mse_loss(style[args.feature_level][0], style[args.feature_level][1], reduction = "none") * weight).mean()
                    # MSE loss
                    elif not args.use_weight: #True, feature_level = 2
                        # Differenza con paper, loss KL calcolata su output del primo layer del decoder invece che su output dell'encoder (che sarebbe c4d)
                        loss_kl = F.mse_loss(style[args.feature_level+1][0], style[args.feature_level+1][1], reduction = "none").mean()
                    else:
                        loss_kl = torch.tensor(0)

                elif args.batch_size == 3:
                    loss_kl = F.mse_loss(style[args.feature_level][0], style[args.feature_level][1])
                    loss_kl += F.mse_loss(style[args.feature_level][1], style[args.feature_level][2])
                    loss_kl += F.mse_loss(style[args.feature_level][0], style[args.feature_level][2])
                    loss_kl /= 6
                    
                # compute gradient and do SGD step
                # if training
                if batch[0]["patient_id"][0] in _labeled_name_list:
                    loss = loss_ + loss_kl * args.weight_kl
                else:
                    loss = loss_kl * args.weight_kl

                epoch_loss = epoch_loss + loss.item()
                    
                loss.backward()  
                optimizer.step()
                
                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                    kl_loss.update(loss_kl.item())
                else:
                    print("NaN in model loss!!")
                
                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)
                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)
                
                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                
                # Display progress
                progress.display(i)

                if args.debug:
                    break
            
            if scheduler is not None:
                scheduler.step()
                    
            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)
            
            ########## log current epoch metrics
            if not args.debug:
                wandb.log({
                    "train/epoch": epoch,
                    "train/seg_loss": losses_.avg,
                    "train/kl_loss": kl_loss.avg,
                    "train/loss": epoch_loss/len(train_loader),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                })

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")

            #Save last model 
            if not args.debug: 
                model_dict = model_1.state_dict()
                save_last(
                    dict(
                        epoch=epoch,
                        state_dict=model_dict,
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        best_1 = best_1
                    ),
                    save_folder=args.save_folder_1)

            # Validate every val check 
            if (epoch+1) in val_check or args.debug:
                #step validation and step of test for our purpose
                #total = 0
                validation_loss, validation_dice, val_metrics = step(val_loader, model_1, args, criterian_val, metric, epoch, t_writer_1,
                                                              save_folder=args.save_folder_1,
                                                              patients_perf=patients_perf, patch_shape = args.patch_shape, debug=args.debug)
                print(validation_dice)
                test_loss, test_dice, test_metrics = step(test_loader, model_1, args, criterian_val, metric, epoch, t_writer_1,
                                            save_folder=args.save_folder_1,
                                            patients_perf=patients_perf, patch_shape = args.patch_shape, debug=args.debug)
                print(test_dice)
                """
                validation_loss_1, validation_dice = step(val_loader, model_1, args, criterian_val, metric, epoch, t_writer_1,
                                                              save_folder=args.save_folder_1,
                                                              patients_perf=patients_perf, mask_modal = [1,3], patch_shape = args.patch_shape)
                total += validation_dice
                print(validation_dice)
                validation_loss_1, validation_dice = step(val_loader, model_1, args, criterian_val, metric, epoch, t_writer_1,
                                                              save_folder=args.save_folder_1,
                                                              patients_perf=patients_perf, mask_modal = [0,1,3], patch_shape = args.patch_shape)
                total += validation_dice
                print(validation_dice)
                validation_loss_1, validation_dice = step(val_loader, model_1, args, criterian_val, metric, epoch, t_writer_1,
                                                              save_folder=args.save_folder_1,
                                                              patients_perf=patients_perf, patch_shape = args.patch_shape)

                total += validation_dice
                print(validation_dice)
                """
                t_writer_1.add_scalar(f"ValidationLoss", validation_loss, epoch)
                t_writer_1.add_scalar(f"ValidationDice", validation_dice, epoch)
                t_writer_1.add_scalar(f"TestLoss", test_loss, epoch)
                t_writer_1.add_scalar(f"TestDice", test_dice, epoch)
                #("ET", "TC", "WT")
                ########## log current epoch metrics
                if not args.debug: # metrics = ("ET", "TC", "WT")
                    wandb.log({
                        "val/loss": validation_loss,
                        "val/dice": validation_dice,
                        "val/epoch": epoch,
                        "val/val_ET_Dice": val_metrics[0],
                        "val/val_WT_Dice": val_metrics[1],
                        "val/val_TC_Dice": val_metrics[2],
                        "test/loss": test_loss,
                        "test/dice": test_dice,
                        "test/epoch": epoch,
                        "test/test_ET_Dice": test_metrics[0],
                        "test/test_WT_Dice": test_metrics[1],
                        "test/test_TC_Dice": test_metrics[2],
                    })
                print(mean_results)

                if validation_dice > best_1:
                    print(f"Saving the model at epoch {epoch} with DSC {validation_dice}")
                    best_1 = validation_dice
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                            best_1 = best_1,
                        ),
                        save_folder=args.save_folder_1)

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break

def step(data_loader, model, args, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None, mask_modal = [], patch_shape = 128, debug=False):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    model.eval()
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        patient_id = val_data["patient_id"]

        #to use a specific missing modal configuration for each sample
        if args.model_type == "vtnet":
            model.module.swin_unet.mask_modal = val_data["mask_modal"]
        else:
            model.module.mask_modal = val_data["mask_modal"]

        with torch.no_grad():
            val_inputs, val_labels = (
                val_data["image"].cuda(),
                val_data["label"].cuda(),
            )
            val_outputs = inference(val_inputs, model, patch_shape = patch_shape) #(1, 3, H, W, Z)
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]

            segs = val_outputs
            targets = val_labels
            loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

        if debug:
            break

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()
    
    #print(metrics)
    metrics = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(dice, device="cpu").numpy()) for dice in metrics]
    labels = ("ET", "TC", "WT")
    #metrics = {key: value for key, value in zip(labels, metrics)}
    #print(metrics)
    return losses.avg, np.nanmean(metrics), metrics


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
