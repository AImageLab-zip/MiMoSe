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
from torch.autograd import Variable
from tensorboardX import SummaryWriter 
import wandb
from dataset.brats import get_datasets_train_rf_forpretrain, get_datasets_brats20_rf, get_datasets23_train_rf_forpretrain
from utils import AverageMeter, ProgressMeter, save_checkpoint, count_parameters, save_args_1, setup_seed, save_last
from model.Unet import Unet_missing
from torch.cuda.amp import autocast as autocast

parser = argparse.ArgumentParser(description='BRATS 2023 Pre-Training')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--mdp', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=600, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--mask_ratio', default=0.875, type=float, help='mask ratio of pretrain')
parser.add_argument('--model_type', type=str, default='vtnet', choices=['vtnet', 'cnnnet'])
parser.add_argument('--dataset', type=str, default='brats23', choices=['brats18', 'brats20', 'brats23'])
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--exp_name', default='patch16mask875_mdp3_inversion_reg2_005', type=str, help='exp name')
parser.add_argument('--val', default=50, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--batch_size', default=1, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=999, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_costum.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--debug', action='store_true', default=False)


def main(args):
    ########## setting seed
    setup_seed(args.seed)

    ########## setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./runs{args.exp_name}/model_1")
    
    ##########init wandb
    if not args.debug:
        slurm_job_id = os.getenv("SLURM_JOB_ID")
        wandb_name_and_id = f'BraTS23_m3ae_pretrain_epoch{args.epochs}_mask_ratio{args.mask_ratio}_jobid{slurm_job_id}'
        wandb.init(
            project="MissingBench",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "M3AE",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "mask_ratio": args.mask_ratio,
                "epochs": args.epochs,
                "mdp": args.mdp,
            }
        )
      
    ########## setting models
    model_1 = Unet_missing(input_shape = [128,128,128], pre_train = True, mask_ratio = args.mask_ratio, mdp = args.mdp) 
    model_1 = nn.DataParallel(model_1)
    print(f"total number of trainable parameters {count_parameters(model_1)}")    
    model_1 = model_1.cuda()

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    ########## setting learning scheduler and optimizer
    limage = []
    ori_para = []
    
    for pname, p in model_1.named_parameters():
        if pname.endswith("limage"):   # limage: modality completion image
            limage.append(p)
        else:
            ori_para.append(p)
    optimizer = torch.optim.Adam([{"params": ori_para}, {"params": limage, "lr": 0.005}], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    ########## setting data
    if args.dataset == "brats18":
        full_train_dataset, l_val_dataset, _, _ = get_datasets_train_rf_forpretrain(args.seed, fold_number=args.fold)
    if args.dataset == "brats20":
        full_train_dataset, l_val_dataset, _, _ = get_datasets_brats20_rf(args.seed, fold_number=args.fold)
    if args.dataset == 'brats23':
        full_train_dataset, l_val_dataset, _, _ = get_datasets23_train_rf_forpretrain(args.seed, fold_number=args.fold)

    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))

    ########## training
    start_epoch = 0
    print("start training now!")

    #########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model_1.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"Reloaded from epoch {checkpoint['epoch']}!")


    for epoch in range(start_epoch, args.epochs): 
        if epoch < args.start_epoch:
            scheduler.step()
            continue
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

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

                inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float() #image = (B, 4, 128, 128, 128), label = (B, 3, 128, 128, 128)
                patch_locations = batch[0]["crop_indexes"]
                
                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                optimizer.zero_grad()
                segs_S1,mask_ratio,_,_= model_1(inputs_S1, patch_locations)

                #MSE Loss
                loss_ = torch.pow((inputs_S1-segs_S1), 2).mean() 
                #Norm-based Loss = Regularization term
                loss2 = torch.norm(model_1.module.limage - model_1.module.limage.mean((2,3,4), keepdim = True), 2).mean() 
                
                loss = loss_ + loss2 * .005
                
                loss.backward()  
                optimizer.step()
                
                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)

                print(f"Loss/{mode}/loss:{loss_.item()}/step:{batch_per_epoch * epoch + i}")
                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)
            
            if scheduler is not None:
                scheduler.step()
            
            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)
            
            ########## log current epoch metrics
            if not args.debug:
                wandb.log({
                    "train/epoch": epoch,
                    "train/loss": losses_.avg,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                })

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            
            #Save last model 
            model_dict = model_1.state_dict()
            save_last(
                dict(
                    epoch=epoch,
                    state_dict=model_dict,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                ),
                save_folder=args.save_folder_1, )
           
            # Save at the end of epoch every val step
            if (epoch + 1) % args.val == 0:                
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder_1, )

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
