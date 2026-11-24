import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload
import pandas as pd
import ast

import numpy as np
import nibabel as nib
import glob
join = os.path.join
import pandas as pd
import random


patch_size = 80
mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join('/path/to/missing_modalities/RFNet', train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()] #875 elements
        # datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1)) # Flatten the segmentation mask
        one_hot_targets = np.eye(self.num_cls)[y] # Convert to one-hot encoding where each voxel is represented as a vector of length num_cls
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1)) # Reshape back to 3D
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))  # Reorder dimensions

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0) #(4)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, test_file='test.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/RFNet', test_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        self.masks = df['mask'].apply(ast.literal_eval)

        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        # target required for models that require the segmentation as one-hot encoded targets, such as Dice loss.
        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y_flatten]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) 

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        #mask = mask_array[index%15]
        mask = np.array(self.masks[index])
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)

        return x, y, mask, yo, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, val_file='val.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/RFNet', val_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']
        #with open(data_file_path, 'r') as f:
        #    datalist = [i.strip() for i in f.readlines()]
        #datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths #125 elements
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        self.masks = df['mask'].apply(ast.literal_eval)

        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        # target required for models that require the segmentation as one-hot encoded targets, such as Dice loss.
        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y_flatten]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) 

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0) #[Channels, Height, Width, Depth]
        y = torch.squeeze(torch.from_numpy(y), dim=0) #[Height, Width, Depth]

        #mask = mask_array[index%15]
        mask = np.array(self.masks[index])
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)

        return x, y, mask, yo, name

    def __len__(self):
        return len(self.volpaths)
