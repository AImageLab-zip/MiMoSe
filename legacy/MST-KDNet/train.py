import yaml
from data import make_data_loaders_divide_2023, make_data_loaders_divide
from models import build_MSTKDNet
from models.discriminator import get_style_discriminator
from solver import make_optimizer_double
from losses import get_losses_divide, bce_loss, get_current_consistency_weight
import os
import torch
import torch.optim as optim
from utils.utils import *
from tqdm import tqdm
import wandb
from utils.random_seed import setup_seed

a=["flair",'t1','t1ce','t2']
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

mask_name = ['t2', 't1', 't1c', 'flair',
            't1t2', 't1cet1', 'flairt1ce', 't1cet2', 'flairt2', 'flairt1',
            'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1cet2',
            'flairt1cet1t2']

val_check = [50, 70, 90, 110, 130, 150, 200, 300, 400, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 800, 825, 850, 900, 910, 920, 930, 940, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000] 
print(f"Validation checks: {val_check}")

def train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, epoch_init=0, best_dice=0.0, pretrain_full_path=None, config=None, logger=None, log_dir=None):
    n_epochs = int(config['epochs'])
    logger.info(f'Start from epoch {epoch_init} to {n_epochs}')
    iter_num = 0

    weight_content = float(config['weight_content'])
    weight_missing = float(config['weight_mispath'])
    weight_full    = 1 - float(config['weight_mispath'])

    for epoch in range(epoch_init, n_epochs):
        weight_consistency = get_current_consistency_weight(epoch)
        scheduler.step()

        train_loss = 0.0
        unnetr_loss_epoch = 0.0
        evd_loss_epoch = 0.0
        slkd_loss_epoch = 0.0
        gsm_loss_epoch = 0.0
        loss_co_epoch = 0.0
        loss_adv_df_trg_main_epoch = 0.0
        loss_d_feature_main_epoch = 0.0
        loss_d_feature_main2_epoch = 0.0

        dice_wt=0.0
        dice_et=0.0
        dice_tc=0.0
        #hd95_wt=0.0
        #hd95_et=0.0
        #hd95_tc=0.0
        #hd95=0.0
        dice=0.0
        
        for phase in ['train', 'val']:
            if phase == 'val' and epoch % 10 != 0:
                continue

            if phase == 'train':
                model_full.train()
                model_missing.train()
                d_style.train()
            else:
                model_full.eval()
                model_missing.eval()
                d_style.eval()

            loader = loaders[phase]
            total = len(loader)

            for batch_id, (batch_x, batch_y, mask) in tqdm(enumerate(loader), total=total, desc="Training Batches"):
                iter_num = iter_num + 1
                batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)

                with torch.set_grad_enabled(phase == 'train'):
                    #rb = random.randint(0, 14)
                    #mask = masks_test[rb]
                    #mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1)
                    mask_tensor = mask.view(1, 4, 1, 1, 1).cuda(non_blocking=True)  # (B, 4, 1, 1, 1)
                    batch_xn = batch_x * mask_tensor                                # (B, 4, 160, 192, 128)

                    seg_f, style_f, content_f,  unetr_fs_f, att_w_f, Gs_f, logit_f = model_full(batch_x[:,0:])
                    seg_m, style_m, content_m, unetr_fs_m, att_w_m, Gs_m, logit_m = model_missing(batch_xn[:,0:])

                    if phase == 'train':
                        # Dice loss + Consistency Term + Logit  Discrepancy  Distillation 
                        loss_dict = losses['co_loss'](config, seg_f, content_f, batch_y, seg_m, content_m, style_f, style_m, epoch)
                        
                        # Feature Distillation
                        unnetr_loss = losses['unetr_loss'](unetr_fs_f, unetr_fs_m)

                        # EVD Loss
                        evd_loss = losses['evd_loss'](att_w_f, att_w_m)

                        # Logit Standardization KL Distillation
                        slkd_loss = losses['slkd_loss'](logit_f, logit_m, temp=7)

                        # Global Style Matching MSE loss
                        gsm_loss = losses['gsm_loss'](Gs_f, Gs_m)

                        loss_dict['loss_Co'] += (unnetr_loss * 0.2 + evd_loss * 1e8 + slkd_loss + gsm_loss * float(config['weight_gsm']))
                        
                        loss_co_epoch += loss_dict['loss_Co'].item()
                        unnetr_loss_epoch += unnetr_loss.item()
                        evd_loss_epoch += evd_loss.item()
                        slkd_loss_epoch += slkd_loss.item()
                        gsm_loss_epoch += gsm_loss.item()

                        print(f"batch-{batch_id}-CoLoss: {loss_dict['loss_Co']}")                            
                        print(f"batch-{batch_id}-full_DiceLoss: {loss_dict['loss_dc']} * {weight_full} = {loss_dict['loss_dc'] * weight_full}")
                        print(f"batch-{batch_id}-missing_DiceLoss: {loss_dict['loss_miss_dc']} * {weight_missing} = {loss_dict['loss_miss_dc'] * weight_missing}")
                        print(f"batch-{batch_id}-consisLoss: {loss_dict['consistency_loss']} * {weight_consistency} = {loss_dict['consistency_loss'] * weight_consistency}")
                        print(f"batch-{batch_id}-content Loss: {loss_dict['content_loss']} * {weight_content} = {loss_dict['content_loss'] * weight_content}")
                        print(f"batch-{batch_id}-unnetrLoss: {unnetr_loss} * 0.2 = {unnetr_loss * (0.2)}")
                        print(f"batch-{batch_id}-EVDLoss: {evd_loss} * 10^8 = {evd_loss * (1e8)}")
                        print(f"batch-{batch_id}-SLKDLoss: {slkd_loss}")
                        print(f"batch-{batch_id}-GSMLoss: {gsm_loss}")
                        
                        #d_style.train()
                        optimizer_d_style = optim.Adam(d_style.parameters(), lr = float(config['lr']), betas=(0.9, 0.99))

                        # labels for style adversarial training
                        source_label = 0
                        target_label = 1

                        optimizer.zero_grad()
                        optimizer_d_style.zero_grad()
                        
                        # only train. Don't accumulate grads in disciminators
                        for param in d_style.parameters():
                            param.requires_grad = False

                        (loss_dict['loss_Co']).backward(retain_graph=True)
                        train_loss += loss_dict['loss_Co'].item()
                    
                        ##################### adversarial training to fool the discriminator ######################
                        df_src_main = style_f
                        df_trg_main = style_m

                        d_df_out_main = d_style(df_trg_main)
                        loss_adv_df_trg_main = bce_loss(d_df_out_main, source_label)
                        loss = 0.0002 * loss_adv_df_trg_main

                        loss_adv_df_trg_main_epoch += loss_adv_df_trg_main.item()

                        loss.backward()                    
                        
                        ####################### Train discriminator networks ######################################
                        ####################### train with multimodal model ##################################################
                        for param in d_style.parameters():
                            param.requires_grad = True

                        df_src_main = df_src_main.detach()
                        d_df_out_main = d_style(df_src_main)
                        loss_d_feature_main = bce_loss(d_df_out_main, source_label)

                        loss_d_feature_main_epoch += loss_d_feature_main.item()

                        loss_d_feature_main.backward()
                
                        ####################### train with missing model ##################################################
                        df_trg_main = df_trg_main.detach()
                        d_df_out_main = d_style(df_trg_main)
                        loss_d_feature_main = bce_loss(d_df_out_main, target_label)

                        loss_d_feature_main2_epoch += loss_d_feature_main.item()

                        loss_d_feature_main.backward()
                
                if phase == 'train':
                    optimizer.step()
                    optimizer_d_style.step()
                    if (batch_id + 1) % 20 == 0:
                        print(f'Epoch {epoch+1}>> itteration {batch_id+1}>> training loss>> {train_loss/(batch_id+1)}')
                else:
                    wt, et, tc = measure_dice_score(seg_m, batch_y, divide=True)
                    dice += (wt + et + tc) / 3.0
                    dice_wt += wt
                    dice_et += et
                    dice_tc += tc

                    #hdwt,hdet,hdtc= measure_hd95(seg_m, batch_y, divide=True)
                    #hd95+=(hdwt+hdet+hdtc)/3.0
                    #hd95_wt+=hdwt
                    #hd95_et+=hdet
                    #hd95_tc+=hdtc

            if phase == 'train':
                # wandb logging
                if not config['debug']:
                    wandb.log({
                        "train/epoch": epoch,
                        "train/loss_co": loss_co_epoch / total,
                        "train/loss_adv_df_trg_main": loss_adv_df_trg_main_epoch / total,
                        "train/loss_d_feature_main": loss_d_feature_main_epoch / total,
                        "train/loss_d_feature_main2": loss_d_feature_main2_epoch / total,
                        "train/unnetr_loss": unnetr_loss_epoch / total,
                        "train/evd_loss": evd_loss_epoch / total,
                        "train/slkd_loss": slkd_loss_epoch / total,
                        "train/gsm_loss": gsm_loss_epoch / total,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    })

                logger.info(f'Epoch {epoch+1} overall training loss>> {train_loss/total}')
                #scheduler.step()

            else:
                dice = (dice / total)
                #hd95 = (hd95/total)
                logger.info(f'Epoch {epoch+1} validation dice score for missing modality whole>> {dice_wt/total} ,core{dice_tc/total},enhance{dice_et/total}')
                #logger.info(f'Epoch {epoch+1} validation hd95 score for missing modality whole>> {hd95_wt/total} ,core{hd95_tc/total},enhance{hd95_et/total}')
                if not config['debug']:
                    wandb.log({
                        "val/epoch": epoch,
                        "val/val_ET_Dice": dice_et / total,
                        "val/val_WT_Dice": dice_wt / total,
                        "val/val_TC_Dice": dice_tc / total,
                        "val/val_Dice": dice,
                    })

                if dice > best_dice:
                    logger.info('save best model ...')
                    state = {}
                    state['model_full'] = model_full.state_dict()
                    state['model_missing'] = model_missing.state_dict()
                    state['d_style'] = d_style.state_dict()
                    state['optimizer'] = optimizer.state_dict()
                    #state['optimizer_d_style'] = optimizer_d_style.state_dict()
                    state['scheduler'] = scheduler.state_dict()
                    state['epochs'] = epoch
                    state['dice'] = dice

                    file_name = log_dir + '/model_best.pth'
                    
                    torch.save(state, file_name)
                    best_dice = dice

        # saving last model
        state = {}
        state['model_full'] = model_full.state_dict()
        state['model_missing'] = model_missing.state_dict()
        state['d_style'] = d_style.state_dict()
        state['optimizer'] = optimizer.state_dict()
        #state['optimizer_d_style'] = optimizer_d_style.state_dict()
        state['scheduler'] = scheduler.state_dict()
        state['epochs'] = epoch
        state['dice'] = best_dice

        file_name = log_dir + '/model_last.pth'

        torch.save(state, file_name)

def main():
    ## Main section    
    ########## print config
    config = yaml.load(open(os.path.join('./configs', ('brats18.yml'))), Loader=yaml.FullLoader)
    for k, v in config.items():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    setup_seed(config["seed"])

    ########## init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'{config["dataset"]}_MST-KDNet_jobid{slurm_job_id}'
    if not config["debug"]:
        wandb.init(
            project="MissingBench",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config=config
        )
    ########## Setting data
    if config['dataset'] == 'brats23':
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits2.csv'
        val_file = 'datalist/val15splits.csv'
        loaders = make_data_loaders_divide_2023(config, train_list_path=train_file, val_list_path=val_file, test_list_path=test_file)
    elif config['dataset'] == 'brats18':
        train_file = 'datalist/Brats18_train.csv'
        test_file = 'datalist/Brats18_test.csv'
        val_file = 'datalist/Brats18_val.csv'
        loaders = make_data_loaders_divide(config, train_list_path=train_file, val_list_path=val_file, test_list_path=test_file)
    else:
        raise NotImplementedError

    
    ########## Setting models 
    model_full, model_missing = build_MSTKDNet(inp_dim1 = 4, inp_dim2 = 4)
    model_full    = model_full.cuda()
    model_missing = model_missing.cuda()
    d_style = get_style_discriminator(num_classes = 128).cuda()
    
    ########## Setting learning schedule and optimizer
    optimizer, scheduler = make_optimizer_double(config, model_full, model_missing)
    # optimizer_d_style = optim.Adam(d_style.parameters(), lr=float(config['lr']), betas=(0.9, 0.99))

    ########## Setting losses
    losses = get_losses_divide(config)
    
    ########## Training
    epoch = 0
    log_dir = os.path.join(config['path_to_log'], config['dataset']+ '_' + str(slurm_job_id))
    latest_model_path = os.path.join(config['path_to_latest_model'], 'model_last.pth')

    if os.path.exists(latest_model_path):
        model_full, model_missing, d_style, optimizer, scheduler, epoch, best_dice = load_old_model(model_full, model_missing, d_style, optimizer, scheduler, latest_model_path)
        epoch = epoch + 1
    else:
        best_dice = 0.0

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  

    logger = get_logging(os.path.join(log_dir, 'train&valid.log'))
    pretrain_full_path = None

    train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, epoch, best_dice, pretrain_full_path, config, logger, log_dir)
    
    #for i in range(0, 15):
    #    test_val(model_missing, loaders, i)
    
    print('Training process is finished')
    if not config["debug"]:
        wandb.finish()
        

if __name__ == '__main__':
    main()
