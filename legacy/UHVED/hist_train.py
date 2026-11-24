"""
Created on Wed Oct 14 17:38:59 2020

@author: reubendo
"""

"""
Adapted from https://github.com/taigw/brats17/blob/master/util/data_process.py

This script renames BRATS dataset to OUTPUT_path,
each subject's images will be cropped and renamed to
"TYPEindex_modality.nii.gz".

output dataset folder will be created if not exists, and content
in the created folder will be, for example:

OUTPUT_path:
   HGG100_Flair.nii.gz
   HGG100_Label.nii.gz
   HGG100_T1c.nii.gz
   HGG100_T1.nii.gz
   HGG100_T2.nii.gz
   ...

Each .nii.gz file in OUTPUT_path will be cropped with a tight bounding box
using function crop_zeros defined in this script.
"""
import os
import nibabel
import numpy as np
import argparse
import pandas as pd 
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
import numpy.ma as ma
from transformations.mean_variance_normalisation import BinaryMaskingLayer
from transformations.histogram_normalisation import HistogramNormalisationLayer
import pandas


mod_names17 = ['flair', 't1', 't1ce', 't2']
mod_names15 = ['Flair', 'T1', 'T1c', 'T2']
mod_names23 = ['t2f', 't1n', 't1c', 't2w']


if __name__ == '__main__':
    """
    Histogram training 
    """
    BRATS_path = '/path/to/missing_modalities/UHVED/extensions/u_hved/datalist/train.txt'
    data_path = '/path/to/missing_modalities/UHVED/data/BRATS2023'
    debug = False
    if not os.path.exists(data_path):
        raise ValueError(
            'Dataset not found: {}'.format(BRATS_path))
    with open(BRATS_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()] #875 elements
    print(f"Length training set: {len(datalist)}")
    
    foreground_masking_layer = BinaryMaskingLayer(
        type_str='mean_plus',
        threshold=0.0) 
    
    histogram_normaliser = HistogramNormalisationLayer(
        image_name='image',
        modalities=mod_names23,
        model_filename='hist.txt',
        binary_masking_func=foreground_masking_layer,
        cutoff=(0.001, 0.999),
        name='hist_norm_layer') 
    
    image_list = []
    count=0
    for pat_folder_name in datalist:
        img_data = []
        for i, mod_i in enumerate(mod_names23):
            img_name = '%s-%s.nii.gz' % (pat_folder_name, mod_i)
            mod_path = os.path.join(data_path, img_name)
            #print(mod_path)
            try:
                mod_data = nibabel.load(mod_path).get_fdata()
                img_data.append(mod_data)
            except OSError:
                print('skipping %s' % mod_path)
                continue
        img_data = np.stack(img_data, axis=-1)
        image_list.append({'image': np.expand_dims(img_data, axis=3)})
        print("subject: {}, shape: {}".format(pat_folder_name, img_data.shape))
        count+=1
        if debug:
            if count==5:
                break
    # Histogram training
    histogram_normaliser.train(image_list=image_list)

  
