from predict import AverageMeter, softmax_output_dice_class4
import glob
import os
import nibabel as nib
import torch

root = "/path/to/nnUNet_raw/Dataset138_BraTS2023/"
dir_list = glob.glob(os.path.join(root, "prediction_*"))

for d in dir_list:
    nifti_list = glob.glob(os.path.join(root, d, "*.nii.gz"))
    vals_evaluation = AverageMeter()
    for f in nifti_list:
        nifti_image = nib.load(f)
        image = nifti_image.get_fdata()
        tensor_image = torch.from_numpy(image)
        f = os.path.basename(f)
        target = nib.load(os.path.join("/path/to/nnUNet_raw/Dataset138_BraTS2023/labelsTs", f)).get_fdata()
        tensor_target = torch.from_numpy(target)
        mask_1 = (image == 1)
        mask_2 = (image == 2)

        # Swap the values
        image[mask_1] = 2
        image[mask_2] = 1

        _, scores_evaluation = softmax_output_dice_class4(torch.unsqueeze(tensor_image, 0), torch.unsqueeze(tensor_target, 0))

        vals_evaluation.update(scores_evaluation)

    output_path = os.path.join(root, d, "dice.txt")
    with open(output_path, 'a') as file:
        file.write(
            'Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n'.format(d,
                                                                                                               vals_evaluation.avg[0][0],
                                                                                                               vals_evaluation.avg[0][1],
                                                                                                               vals_evaluation.avg[0][2]))





