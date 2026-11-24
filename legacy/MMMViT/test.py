import torch
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader 
import mmmvit
from argparse import ArgumentParser
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datapath',type=str,required=True)
    parser.add_argument('--resume',type=str,required=True)
    parser.add_argument('--savepath',type=str,default=f"eval")

    args=parser.parse_args()
    datapath = args.datapath
    resume = args.resume
    savepath = args.savepath

    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
    
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    #datapath = '/path/to/missing_modalities/BRATS2023_Training_npy'
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datalist/test15splits2.csv')
    #resume = '/path/to/missing_modalities/mmFormer/output/best.pth'
    num_cls = 4
    dataname = 'BRATS2025'

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = mmmvit.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch'] + 1
    os.makedirs(savepath,exist_ok=True)
    flipped = True if 'flipped' in datapath else False
    output_path = os.path.join(savepath, f"output_{dataname}_{flipped if flipped else ''}_{best_epoch}.txt")

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))
