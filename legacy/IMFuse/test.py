import torch
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader 
from IMFuse import IMFuse
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--test_file', default='datalist/test15splits2.csv', type=str)
parser.add_argument('--datapath', default="/path/to/missing_modalities/BRATS2023_Training_npy", type=str)
parser.add_argument('--interleaved_tokenization', action='store_true', default=False)
parser.add_argument('--mamba_skip', action='store_true', default=False)
#parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

if __name__ == '__main__':
    args = parser.parse_args()
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
    
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = args.datapath
    test_file = args.test_file
    save_path = args.savepath
    num_cls = 4
    dataname = args.dataname
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = IMFuse(
                num_cls=num_cls,
                interleaved_tokenization=args.interleaved_tokenization,
                mamba_skip=args.mamba_skip
            )
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch'] + 1
    out_path = args.savepath
    output_path = f"{out_path}_{best_epoch}_{index}.txt"

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks[index*5:(index+1)*5]):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False,
                            save_masks=True,
                            save_dir=save_path,
                            index = index)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))
