import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange, RandCrop3D_Loc
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

patch_size = 128

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True],
                      [True, True, False, False], [True, False, True, False], [True, False, False, True], [False, True, True, False], [False, True, False, True], [False, False, True, True], [True, True, True, False], [True, True, False, True], [True, False, True, True], [False, True, True, True],
                      [True, True, True, True]])

def get_crop_slice(target_size, dim):
    # dim is the ori shape
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)
    
def pad_or_crop_image(image, seg=None, target_size=(128, 128, 128), indices=None):
    c, x, y, z= image.shape
    # Generate slices for cropping based on target size
    x_slice, y_slice, z_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (x, y, z))]
    #print(z_slice[0], z_slice[1], z_slice[2])
    #print(z_slice)
    xmin, ymin, zmin = [int(arr.start) for arr in (x_slice, y_slice, z_slice)]
    xmax, ymax, zmax = [int(arr.stop) for arr in (x_slice, y_slice, z_slice)]
    crop_indexes=[[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    
    # Apply slicing 
    image = image[:, x_slice, y_slice, z_slice]
    # If segmentation exists, apply the same slice to it
    if seg is not None:
        seg = seg[x_slice, y_slice, z_slice]
    
    # Calculate padding for each dimension z, y, x
    #todos = [get_left_right_idx_should_pad(size, dim, index, dimage) for size, dim, index, dimage in zip(target_size, [z, y, x], indices, (27, 112, 112))]
    #padlist = [(0, 0)]  # channel dim
    #for idx, to_pad in enumerate(todos):
    #    if to_pad[0]:
    #        padlist.append((to_pad[1], to_pad[2]))
    #        crop_indexes[idx][0] -= to_pad[1]
    #        crop_indexes[idx][1] += to_pad[2]
    #    else:
    #        padlist.append((0, 0))
    # Apply padding 
    #image = np.pad(image, padlist)
    #if seg is not None:
    #    seg = np.pad(seg, padlist)
    return image, seg, crop_indexes
    #return image, crop_indexes

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join('/path/to/missing_modalities/mmFormer/mmformer', train_file)
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

class Brats_loadall_nii_loc(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join('/path/to/missing_modalities/mmFormer/mmformer', train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()] #875 elements
        # datalist.sort()

        #read location files
        self.df = pd.read_csv("/path/to/missing_modalities/crop_indices.csv")

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
        #x, y = x[None, ...], y[None, ...]

        #read location of the case_id
        row = self.df[self.df["case_id"] == name]
        if not row.empty:
            x_min, x_max = row["x_min"].values[0], row["x_max"].values[0]
            y_min, y_max = row["y_min"].values[0], row["y_max"].values[0]
            z_min, z_max = row["z_min"].values[0], row["z_max"].values[0]
        else:
            print("case_id not found.")

        x, y, idx2 = pad_or_crop_image(x.transpose(3, 0, 1, 2), y, target_size=(patch_size, patch_size, patch_size), indices=(z_min, y_min, x_min))
        
        #update bounding box
        #idx2 = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        x_min += idx2[0][0]
        y_min += idx2[1][0]
        z_min += idx2[2][0]
        
        x_max = x_min + patch_size
        y_max = y_min + patch_size
        z_max = z_min + patch_size

        #augmentations
        x, y = x.transpose(1, 2, 3, 0)[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))    # [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1)) 
        one_hot_targets = np.eye(self.num_cls)[y] 
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1)) 
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))  

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0) #(4)

        return x, yo, mask, x_min, x_max, y_min, y_max, z_min, z_max, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, test_file='test.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/mmFormer/mmformer', test_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']
        #with open(data_file_path, 'r') as f:
        #    datalist = [i.strip() for i in f.readlines()] #251 elements
        #datalist.sort()
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
        data_file_path = os.path.join('/path/to/missing_modalities/mmFormer/mmformer', val_file)
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

class Brats_loadall_val_nii_loc(Dataset):
    def __init__(self, transforms='', root=None, val_file='val.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/mmFormer/mmformer', val_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']

        #read location files
        self.df = pd.read_csv("/path/to/missing_modalities/crop_indices.csv")

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

        #read location of the case_id
        row = self.df[self.df["case_id"] == name]

        if not row.empty:
            x_min, x_max = row["x_min"].values[0], row["x_max"].values[0]
            y_min, y_max = row["y_min"].values[0], row["y_max"].values[0]
            z_min, z_max = row["z_min"].values[0], row["z_max"].values[0]
        else:
            print("case_id not found.")

        return x, y, mask, yo, x_min, x_max, y_min, y_max, z_min, z_max, name
    
    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii_loc(Dataset):
    def __init__(self, transforms='', root=None, test_file='test.txt', modal='all', num_cls=4):
        data_file_path = os.path.join('/path/to/missing_modalities/mmFormer/mmformer', test_file)
        df = pd.read_csv(data_file_path)
        datalist = df['case']
        
        #read location files
        self.df = pd.read_csv("/path/to/missing_modalities/crop_indices.csv")

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

        #read location of the case_id
        row = self.df[self.df["case_id"] == name]

        if not row.empty:
            x_min, x_max = row["x_min"].values[0], row["x_max"].values[0]
            y_min, y_max = row["y_min"].values[0], row["y_max"].values[0]
            z_min, z_max = row["z_min"].values[0], row["z_max"].values[0]
        else:
            print("case_id not found.")

        return x, y, mask, yo, x_min, x_max, y_min, y_max, z_min, z_max, name
            
    def __len__(self):
        return len(self.volpaths)
