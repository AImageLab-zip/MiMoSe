import argparse
import sys
sys.path.append("..")
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from utils.random_seed import setup_seed
import os
import os.path as osp
from DualNet_SS import DualNet_SS as DualNet
from BraTSDataSet import BraTSDataSet, BraTSValDataSet, my_collate
import timeit
from tensorboardX import SummaryWriter
import loss_Dual as loss
from engine import Engine
from math import ceil
import wandb
import random

from DualNet_SS import conv3x3x3

start = timeit.default_timer()
alpha = 0.1  # shared domain loss weight
beta = 0.02  # specific domain loss weight
calc_flops = False
val_check_wu = [15000, 30000, 65000, 100000, 135000, 170000, 205000, 220000, 235000, 240000, 245000] 
val_check_train = [110000, 120000, 130000, 150000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 250000, 270000, 290000, 310000, 320000, 330000, 340000, 350000, 360000, 370000, 380000, 390000, 406000] 

print(f"Validation checks warm up: {val_check_wu}")
print(f"Validation checks training: {val_check_train}")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--train_list", type=str, default='BraTS20/BraTS20_train.csv')
    parser.add_argument("--val_list", type=str, default='BraTS20_val.csv')
    parser.add_argument("--test_list", type=str, default='BraTS20_test.csv')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/example/')
    parser.add_argument("--reload_path", type=str, default='snapshots/example/last.pth')
    parser.add_argument("--reload_from_checkpoint", action='store_true', default=False) #parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=875)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--weight_std", action='store_true', default=True) #parser.add_argument("--weight_std", type=str2bool, default=True) 
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", action='store_true', default=False) #parser.add_argument("--random_mirror", type=str2bool, default=False)
    parser.add_argument("--random_scale", action='store_true', default=False) #parser.add_argument("--random_scale", type=str2bool, default=False)
    parser.add_argument("--random_seed", type=int, default=999)

    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--warm_up", action="store_true")
    parser.add_argument("--mode", type=str, default='0,1,2,3')

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()


def compute_dice_score(preds, labels):
    preds = F.sigmoid(preds)

    pred_ET = preds[:, 0, :, :, :]
    pred_WT = preds[:, 1, :, :, :]
    pred_TC = preds[:, 2, :, :, :]
    label_ET = labels[:, 0, :, :, :]
    label_WT = labels[:, 1, :, :, :]
    label_TC = labels[:, 2, :, :, :]
    dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
    dice_WT = dice_score(pred_WT, label_WT).cpu().data.numpy()
    dice_TC = dice_score(pred_TC, label_TC).cpu().data.numpy()

    """
    pred_NCR = preds[:, 0, :, :, :]
    pred_ED = preds[:, 1, :, :, :]
    pred_ET = preds[:, 2, :, :, :]

    label_NCR = labels[:, 0, :, :, :]
    label_ED = labels[:, 1, :, :, :]
    label_ET = labels[:, 2, :, :, :]

    dice_NCR = dice_score(pred_NCR, label_NCR).cpu().data.numpy()
    dice_ED = dice_score(pred_ED, label_ED).cpu().data.numpy()
    dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
    """

    return dice_ET, dice_WT, dice_TC


def predict_sliding(args, net, imagelist, mask_indices, tile_size, classes):
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction, _, _, _ = net(img, mask_indices, val=True, mode=args.mode)

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate(args, input_size, model, ValLoader, num_classes, loss_D, loss_BCE):
    # start to validate
    val_ET = 0.0
    val_WT = 0.0
    val_TC = 0.0
    loss_seg = 0.0

    for index, batch in enumerate(ValLoader):
        image, image_res, label, mask, mask_indices, size, name, affine = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        mask_indices = mask_indices[0]
        #mask = mask.cuda()

        with torch.no_grad():
            pred = predict_sliding(args, model, [image, image_res], mask_indices, input_size, num_classes)
            
            term_seg_Dice = loss_D.forward(pred, label)
            term_seg_BCE = loss_BCE.forward(pred, label)
            term_seg = term_seg_Dice + term_seg_BCE

            dice_ET, dice_WT, dice_TC = compute_dice_score(pred, label)

            val_ET += dice_ET
            val_WT += dice_WT
            val_TC += dice_TC
            loss_seg += term_seg

    return val_ET/(index+1), val_WT/(index+1), val_TC/(index+1), loss_seg/(index+1)


def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    print(parser)
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    print(torch.cuda.is_available())

    with Engine(custom_parser=parser) as engine:
        ########## setting 
        args = parser.parse_args()
        wandb_name_and_id = f'BraTS23_ShaSpec_steps{args.num_steps}_{"warmup" if args.warm_up else ""}_jobid{slurm_job_id}'
        wandb.init(
            project="MissingBench",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "ShaSpec",
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "num_steps": args.num_steps,
                "train_list": args.train_list,
                "val_list": args.val_list,
                "normalization": args.norm_cfg,
                "activation": args.activation_cfg,
                "train_only": args.train_only,
                "mode": args.mode,
                "warm_up": args.warm_up,
            }
        )
        for k, v in args._get_kwargs():
            pad = ' '.join(['' for _ in range(25-len(k))])
            print(f"{k}:{pad} {v}", flush=True)
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        snapshot_dir = args.snapshot_dir #+ f'.{slurm_job_id}/'
        writer = SummaryWriter(snapshot_dir)
        print('current mode:', args.mode)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)

        ########## setting seed
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        setup_seed(seed)
        
        ########## setting model
        model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                        num_classes=args.num_classes, weight_std=args.weight_std, self_att=False, cross_att=False)

        if calc_flops:
            from thop import profile
            input = torch.randn(1, 4, 80, 160, 160)
            macs, params = profile(model, inputs=(input,))
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return
        model.train()
        device = torch.device('cuda:{}'.format(args.local_rank))
        model.to(device)

        ########## setting learning scheduler, optimizer and criterions
        # optimizer = optim.Adam(
        #     [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
        #     lr=args.learning_rate, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)
        # optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
        #                                                        patience=args.patience, verbose=True, threshold=1e-3,
        #                                                        threshold_mode='abs')

        if args.num_gpus > 1:
            model = engine.data_parallel(model)

        loss_D = loss.DiceLoss4BraTS().to(device)
        loss_BCE = loss.BCELoss4BraTS().to(device)
        loss_domain_cls = loss.DomainClsLoss().to(device)
        distribution_loss = nn.L1Loss()
        # distribution_loss = nn.MSELoss()
        # distribution_loss = nn.KLDivLoss(reduction='sum')
        # model.kl_prj = conv3x3x3(256, 1, kernel_size=3, padding=0, bias=False, weight_std=args.weight_std).to(device)

        ########## setting data
        trainloader, train_sampler = engine.get_train_loader(BraTSDataSet(args.data_dir, args.train_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror), collate_fn=my_collate)
        valloader, val_sampler = engine.get_test_loader(BraTSValDataSet(args.data_dir, args.val_list))
        testloader, test_sampler = engine.get_test_loader(BraTSValDataSet(args.data_dir, args.test_list))

        ########## training
        val_Dice_best = -999999
        loss_iter = 0.0
        shared_loss_iter = 0.0
        seg_loss_iter = 0.0
        spec_loss_iter = 0.0
        count = 0

        ########## load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                # model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
                checkpoint = torch.load(args.reload_path)
                model = checkpoint['model']
                optimizer = checkpoint['optimizer']
                args.start_iters = checkpoint['iter'] + 1
                val_Dice_best = checkpoint['val_Dice_best'] 
                print("Loaded model trained for", args.start_iters, "iters")
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))
                exit(0)

        for i_iter, batch in enumerate(trainloader):
            i_iter += args.start_iters
            images = torch.from_numpy(batch['image']).cuda()
            labels = torch.from_numpy(batch['label']).cuda()

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

            iter_start = timeit.default_timer()
            preds, shared_info, spec_info, mode_split = model(images, mode=args.mode)
            iter_end = timeit.default_timer()
            # print(iter_end - iter_start, 'seconds')
            preds_seg = preds

            # segmentation loss
            term_seg_Dice = loss_D.forward(preds_seg, labels)
            term_seg_BCE = loss_BCE.forward(preds_seg, labels)

            N, C, D, W, H = images.shape

            # SimShareAllPairs 
            term_shared = distribution_loss(shared_info[0:N * C + 1:C], shared_info[1:N * C + 1:C]) + \
                          distribution_loss(shared_info[1:N * C + 1:C], shared_info[2:N * C + 1:C]) + \
                          distribution_loss(shared_info[2:N * C + 1:C], shared_info[3:N * C + 1:C]) + \
                          distribution_loss(shared_info[3:N * C + 1:C], shared_info[0:N * C + 1:C]) #+ \
                          # distribution_loss(shared_info[0:N * C + 1:C], shared_info[2:N * C + 1:C]) + \
                          # distribution_loss(shared_info[1:N * C + 1:C], shared_info[3:N * C + 1:C])

            # specific domain loss
            # flair, t1, t1ce , t2 are domain 0, 1, 2, 3, respectively
            spec_labels = torch.zeros(N * C, dtype=torch.long).to(device) #(0, 1, 2, 3)
            spec_labels[0:N * C + 1:C] = 0
            spec_labels[1:N * C + 1:C] = 1
            spec_labels[2:N * C + 1:C] = 2
            spec_labels[3:N * C + 1:C] = 3
            term_spec = loss_domain_cls.forward(spec_info, spec_labels)

            term_all = term_seg_Dice + term_seg_BCE + alpha * term_shared + beta * term_spec
            term_all.backward()

            optimizer.step()

            loss_iter += term_all
            shared_loss_iter += term_shared
            seg_loss_iter += (term_seg_Dice+term_seg_BCE)
            spec_loss_iter += term_spec
            count+=1

            print('iter = {} of {} completed, lr = {:.4}, seg_loss = {:.4}, shared_loss = {:.4}, spec_loss = {:.4}'
                .format(i_iter, args.num_steps, lr, (term_seg_Dice+term_seg_BCE).cpu().data.numpy(),
                        term_shared.cpu().data.numpy(), term_spec.cpu().data.numpy()))

            #end of an "epoch": logging
            if i_iter % 875 == 0 and (args.local_rank == 0) and (i_iter != 0):
                wandb.log({
                    "train/iter": i_iter+1,
                    "train/learning_rate": lr,
                    "train/loss": loss_iter.cpu().detach().item() / count,
                    "train/shared_loss": shared_loss_iter.cpu().detach().item()  / count,
                    "train/seg_loss": seg_loss_iter.cpu().detach().item() / count,
                    "train/spec_loss": spec_loss_iter.cpu().detach().item() / count,
                })
                loss_iter = 0.0
                shared_loss_iter = 0.0
                seg_loss_iter = 0.0
                spec_loss_iter = 0.0
                count=0

                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', term_all.cpu().data.numpy(), i_iter)

            #save last iter model
            if i_iter >= args.num_steps - 1 and (args.local_rank == 0):
                print('save last model ...')
                checkpoint = {
                    'model': model,
                    'optimizer': optimizer,
                    'iter': i_iter,
                    'val_Dice_best': val_Dice_best,
                }
                torch.save(checkpoint, osp.join(snapshot_dir, 'final.pth'))
                
            #save every val_pred_every iters = 1 epoch
            if (i_iter % args.val_pred_every == args.val_pred_every - 1) and i_iter != 0 and (args.local_rank == 0):
                print(f'save model {i_iter} ...')
                checkpoint = {
                    'model': model,
                    'optimizer': optimizer,
                    'iter': i_iter,
                    'val_Dice_best': val_Dice_best,
                }
                # torch.save(checkpoint, osp.join(snapshot_dir, 'iter_' + str(i_iter) + '.pth'))
                torch.save(checkpoint, osp.join(snapshot_dir, 'last.pth'))

            # validation and testing
            if args.warm_up:
                val_check = val_check_wu
            else:
                val_check = val_check_train

            if i_iter+1 in val_check:
                model.eval()

                print('validate ...')
                val_ET, val_WT, val_TC, seg_loss = validate(args, input_size, model, valloader, args.num_classes, loss_D, loss_BCE)
                wandb.log({
                    "val/iter":i_iter+1,
                    "val/val_ET_Dice": val_ET,
                    "val/val_WT_Dice": val_WT,
                    "val/val_TC_Dice": val_TC,
                    "val/val_Dice": (val_ET + val_WT + val_TC)/3,
                    "val/seg_loss": seg_loss.cpu().detach().item(),       
                })
                
                if (args.local_rank == 0):
                    writer.add_scalar('Val_ET_Dice', val_ET, i_iter)
                    writer.add_scalar('Val_WT_Dice', val_WT, i_iter)
                    writer.add_scalar('Val_TC_Dice', val_TC, i_iter)
                    print('Validate iter = {}, ET = {:.2}, WT = {:.2}, TC = {:.2}'.format(i_iter, val_ET, val_WT, val_TC))

                if i_iter!=0 and (args.local_rank == 0) and (val_ET + val_WT + val_TC)/3 > val_Dice_best:
                    val_Dice_best = (val_ET + val_WT + val_TC)/3
                    print('save model ...')
                    checkpoint = {
                        'model': model,
                        'optimizer': optimizer,
                        'iter': i_iter,
                        'val_Dice_best': val_Dice_best,
                    }
                    torch.save(checkpoint, osp.join(snapshot_dir, 'best.pth'))
                
                print('testing ...')
                test_ET, test_WT, test_TC, seg_loss = validate(args, input_size, model, testloader, args.num_classes, loss_D, loss_BCE)
                wandb.log({
                    "test/iter":i_iter+1,
                    "test/test_ET:_Dice": test_ET,
                    "test/test_WT_Dice": test_WT,
                    "test/test_TC_Dice": test_TC,
                    "test/test_Dice": (test_ET + test_WT + test_TC)/3,
                    "test/seg_loss": seg_loss.cpu().detach().item(),       
                })
                
                if (args.local_rank == 0):
                    writer.add_scalar('test_ET_Dice', test_ET, i_iter)
                    writer.add_scalar('test_WT_Dice', test_WT, i_iter)
                    writer.add_scalar('test_TC_Dice', test_TC, i_iter)
                    print('Testing iter = {}, ET = {:.2}, WT = {:.2}, TC = {:.2}'.format(i_iter, test_ET, test_WT, test_TC))

                model.train()

    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
