import multiprocessing
import os.path
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == '__main__':
    brats_data_dir = '/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'

    task_id = 138
    task_name = "BraTS2023"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(out_base)
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)

    for c in case_ids:
        if not os.path.exists(join(labelstr, c + '.nii.gz')):
            shutil.copy(join(brats_data_dir, c, c + "-t2w.nii.gz"), join(imagestr, c + '_0000.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "-t1c.nii.gz"), join(imagestr, c + '_0001.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "-t1n.nii.gz"), join(imagestr, c + '_0002.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "-t2f.nii.gz"), join(imagestr, c + '_0003.nii.gz'))
            shutil.copy(join(brats_data_dir, c, c + "-seg.nii.gz"), join(labelstr, c + '.nii.gz'))

        # copy_BraTS_segmentation_and_convert_labels_to_nnUNet(join(brats_data_dir, c, c + "_seg.nii.gz"),
        #                                                     join(labelstr, c + '.nii.gz'))

    # generate_dataset_json(out_base,
    #                       channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
    #                       labels={
    #                           'background': 0,
    #                           'whole tumor': (1, 2, 3),
    #                           'tumor core': (2, 3),
    #                           'enhancing tumor': (3, )
    #                       },
    #                       num_training_cases=len(case_ids),
    #                       file_ending='.nii.gz',
    #                       regions_class_order=(1, 2, 3),
    #                       license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
    #                       reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
    #                       dataset_release='1.0')

    generate_dataset_json(out_base,
                          channel_names={0: 'T2', 1: 'T1c', 2: 'T1', 3: 'Flair'},
                          labels={
                              'background': 0,
                              'whole tumor': (1, 2, 3),
                              'tumor core': (1, 3),
                              'enhancing tumor': (3, )
                          },
                          num_training_cases=len(case_ids),
                          file_ending='.nii.gz',
                          regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/Synapse:syn51156910/wiki/622351',
                          reference='see https://www.synapse.org/Synapse:syn51156910/wiki/622351',
                          dataset_release='1.0')