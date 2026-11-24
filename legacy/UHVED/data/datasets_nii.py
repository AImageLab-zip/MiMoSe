import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad, RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from transformations.rand_spatial_scaling import RandomSpatialScalingLayer
from transformations.rand_flip import RandomFlipLayer
from transformations.rand_rotation import RandomRotationLayer
from .data_utils import pkload
import pandas as pd
import ast

import numpy as np
import nibabel as nib
import glob
join = os.path.join

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])
MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']

def get_masking():
    choices = []
    nb_choices = np.random.randint(4)
    choices = np.random.choice(4, nb_choices+1, replace=False, p=[1/4,1/4,1/4,1/4]) #Randomly selects nb_choices + 1 unique indices from the range [0, 1, 2, 3]
    choices = [True if k in choices else False for k in range(4)]
    return choices


class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join('/path/to/missing_modalities/UHVED', train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()] #875 elements
        
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        """
        self.files = []
        for dataname in datalist:
            flair_file = dataname + '-t2f.nii.gz'  #BraTS23
            t1_file = dataname + '-t1n.nii.gz'
            t1ce_file = dataname + '-t1c.nii.gz'
            t2_file = dataname + '-t2w.nii.gz'
            label_file = dataname + '-seg.nii.gz'
            self.files.append({
                "flair": os.path.join(root, flair_file),
                "t1": os.path.join(root, t1_file),
                "t1ce": os.path.join(root, t1ce_file),
                "t2": os.path.join(root, t2_file),
                "label": os.path.join(root, label_file),
                "name": dataname
            })

        self.flip_layer = RandomFlipLayer(flip_axes=(0,1,2))
        self.flip_layer.randomise()
        self.scaling_layer = RandomSpatialScalingLayer(
            min_percentage=-10.0,  # Minimum scaling factor as a percentage
            max_percentage=10.0,   # Maximum scaling factor as a percentage
        )
        self.scaling_layer.randomise()
        self.rotation_layer = RandomRotationLayer()
        self.rotation_layer.init_uniform_angle((-10, 10))
        self.rotation_layer.randomise()
        """
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls

    def __getitem__(self, index):
        """
        datafiles = self.files[index]
        t1 = nib.load(datafiles["t1"]).get_fdata()
        t1ce = nib.load(datafiles["t1ce"]).get_fdata()
        t2 = nib.load(datafiles["t2"]).get_fdata()
        flair = nib.load(datafiles["flair"]).get_fdata()
        y = nib.load(datafiles["label"]).get_fdata()
        x = np.stack([t1, t1ce, t2, flair], axis=-1).astype(np.float32)
        """
        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...] #(1, H, W, T, 4)

        ### Augmentations
        """
        x = np.transpose(x, (1, 2, 3, 0, 4)) #(H, W, T, 1, 4)
        #flip axes 
        x = self.flip_layer(x)
        #random scaling
        input = {'image': x}
        interp_orders = {'image': [3]}
        x = self.scaling_layer(input, interp_orders)
        #random rotation
        x = self.rotation_layer(input, interp_orders)['image']
        x = np.transpose(x, (3, 0, 1, 2, 4)) #(1, H, W, T, 4)
        #crop
        #x = self.transforms(x).astype(np.float32) #(1, 112, 112, 112, 4)
        #y = self.transforms(y).astype(np.int64) #(1, 112, 112, 112)
        """
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3)) # [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1)) # Flatten the segmentation mask
        one_hot_targets = np.eye(self.num_cls)[y_flatten] # Convert to one-hot encoding where each voxel is represented as a vector of length num_cls
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1)) # Reshape back to 3D
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))  # Reorder dimensions
        
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        #y = torch.squeeze(torch.from_numpy(y), dim=0)
        
        #mask = get_masking() #np.random.choice(15, 1)
        #mask = torch.tensor(mask)  #(4)
        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0) #(4)
        return x, yo, mask, name #, y

    def __len__(self):
        return len(self.volpaths) #len(self.files)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, test_file='test.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/UHVED', test_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        """
        self.files = []
        for dataname in datalist:
            flair_file = dataname + '-t2f.nii.gz'  #BraTS23
            t1_file = dataname + '-t1n.nii.gz'
            t1ce_file = dataname + '-t1c.nii.gz'
            t2_file = dataname + '-t2w.nii.gz'
            label_file = dataname + '-seg.nii.gz'
            self.files.append({
                "flair": os.path.join(root, flair_file),
                "t1": os.path.join(root, t1_file),
                "t1ce": os.path.join(root, t1ce_file),
                "t2": os.path.join(root, t2_file),
                "label": os.path.join(root, label_file),
                "name": dataname
            })
        """
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        self.masks = df['mask'].apply(ast.literal_eval)

    def __getitem__(self, index):
        """
        datafiles = self.files[index]
        t1 = nib.load(datafiles["t1"]).get_fdata()
        t1ce = nib.load(datafiles["t1ce"]).get_fdata()
        t2 = nib.load(datafiles["t2"]).get_fdata()
        flair = nib.load(datafiles["flair"]).get_fdata()
        y = nib.load(datafiles["label"]).get_fdata()
        """
        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        #x = np.stack([t1, t1ce, t2, flair], axis=-1)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        # target required for models that require the segmentation as one-hot encoded targets, such as Dice loss.
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y_flatten]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) 

        y = np.ascontiguousarray(y)

        #x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        #mask = mask_array[index%15]
        mask = np.array(self.masks[index])
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)

        return x, y, mask, yo, name

    def __len__(self):
        return len(self.volpaths)
        #return len(self.files)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, val_file='val.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/UHVED', val_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths #125 elements
        """
        self.files = []
        for dataname in datalist:
            flair_file = dataname + '-t2f.nii.gz'  #BraTS23
            t1_file = dataname + '-t1n.nii.gz'
            t1ce_file = dataname + '-t1c.nii.gz'
            t2_file = dataname + '-t2w.nii.gz'
            label_file = dataname + '-seg.nii.gz'
            self.files.append({
                "flair": os.path.join(root, flair_file),
                "t1": os.path.join(root, t1_file),
                "t1ce": os.path.join(root, t1ce_file),
                "t2": os.path.join(root, t2_file),
                "label": os.path.join(root, label_file),
                "name": dataname
            })#125 elements
        """
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        self.masks = df['mask'].apply(ast.literal_eval)

    def __getitem__(self, index):
        """
        datafiles = self.files[index]
        t1 = nib.load(datafiles["t1"]).get_fdata()
        t1ce = nib.load(datafiles["t1ce"]).get_fdata()
        t2 = nib.load(datafiles["t2"]).get_fdata()
        flair = nib.load(datafiles["flair"]).get_fdata()
        y = nib.load(datafiles["label"]).get_fdata()
        x = np.stack([t1, t1ce, t2, flair], axis=-1)
        """
        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        #x = x.astype(np.float32)
        #y = y.astype(np.int64)

        # target required for models that require the segmentation as one-hot encoded targets, such as Dice loss.
        _, H, W, Z = np.shape(y)
        y_flatten = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y_flatten]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        yo = torch.squeeze(torch.from_numpy(yo), dim=0) 

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        #x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0) #[Channels, Height, Width, Depth]
        y = torch.squeeze(torch.from_numpy(y), dim=0) #[Height, Width, Depth]

        #mask = mask_array[index%15]
        mask = np.array(self.masks[index])
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)

        return x, y, mask, yo, name

    def __len__(self):
        return len(self.volpaths)#len(self.files)
