import pathlib
import SimpleITK as sitk
import numpy as np
import torch
import glob
import random
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Dataset
from pathlib import Path
import os
import csv
from dataset.image_utils import pad_or_crop_image, irm_min_max_preprocess, zscore_normalise
from dataset.transforms import *
import pandas as pd
import ast 

user = "LH"
BRATS_TRAIN_FOLDERS='/yourpath/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Training'
TEST_FOLDER='/yourpath/2-MICCAI_BraTS_2018/MICCAI_BraTS_2018_Data_Validation'
BRATS_TRAIN_FOLDERS_20='Path of your brats dataset'  # (*/BRATS2020_Training_none_npy)
BRATS_TRAIN_FOLDERS_23='/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'  # (*/BRATS2020_Training_none_npy)
ROOT = '/path/to/missing_modalities/m3ae/' #root of the project

def get_brats_folder():
    return BRATS_TRAIN_FOLDERS

def get_brats_folder_20():
    return BRATS_TRAIN_FOLDERS_20

def get_brats_folder_23():
    return BRATS_TRAIN_FOLDERS_23

def get_test_brats_folder():
    return TEST_FOLDER


class Brats(Dataset):
    def __init__(self, patients_dir, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax", transforms = None, patch_shape = 128):
        super(Brats, self).__init__()
        self.patch_shape = patch_shape
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.transforms = eval(transforms or 'Identity()')
        self.patterns = ["-t1n", "-t1c", "-t2w", "-t2f"]
        if not no_seg:
            self.patterns += ["-seg"]
        for patient_dir in patients_dir:
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        
        if patient_image.shape != (4, 155, 240, 240):
            print(f"Different shape: {patient_image.shape}")
        
        if _patient["seg"] is not None:
            et = patient_label == 3
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 3, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            # Indices of non-zero elements
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Calculates the minimum and maximum indices for each dimension (z, y, x) and Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label, idx2 = pad_or_crop_image(patient_image, patient_label, target_size=(self.patch_shape, self.patch_shape, self.patch_shape), indices=(zmin, ymin, xmin))
            # Updates the bounding box 
            zmin += idx2[0][0]
            ymin += idx2[1][0]
            xmin += idx2[2][0]

            zmax = zmin + self.patch_shape
            ymax = ymin + self.patch_shape
            xmax = xmin + self.patch_shape

            #if zmin<0 or zmax>155:
            #   print(f"Wrong coordinates for z:{(zmin, zmax)}")

            #print(f"Bounding box after threshold: ({zmin}, {zmax}), ({ymin}, {ymax}), ({xmin}, {xmax})")       
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if not self.training:
            patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
            patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label,
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    idx=idx,
                    )

    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)


def get_datasets_train_rf_forpretrain(seed, on="train", fold_number=0, normalisation="minmax", part = 1, all_data = True, patch_shape = 128):

    # use training set & val set for pretrain.

    data_root = get_brats_folder()
    
    data_file_path = "dataset/train3.txt"
    #train_list = []
    with open(data_file_path, 'r') as f:
        train_list = [i.strip() for i in f.readlines()]
    train_list.sort()
    
    data_file_path = "dataset/val3.txt"
    #train_list = []
    with open(data_file_path, 'r') as f:
        train_list.extend([i.strip() for i in f.readlines()])
    train_list.sort()
    
    data_file_path = "dataset/test3.txt"
    with open(data_file_path, 'r') as f:
        test_list = [i.strip() for i in f.readlines()]
    test_list.sort()
    
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    train = []
    test = []
    for l in patients_dir:
        for l2 in train_list:
            if l2 in l:
                train.append(l)
                break
    
    for l in patients_dir:
        for l2 in test_list:
            if l2 in l:
                test.append(l)
                break
                
    train = [Path(l) for l in train]
    test = [Path(l) for l in test]
    
    train_f = open("train.txt", "w")
    csv_writer = csv.writer(train_f)

    print("train length: " , len(train))
    print("val length: " , len(test))
    
    if part != 1:
        sub_train = train[:int(len(train) * part)+1]
        if not all_data:
            train = sub_train

    # return patients_dir

    train_dataset = Brats(train, training=True,
                          normalisation=normalisation, patch_shape = patch_shape)
    train_dataset2 = Brats(train, training=False, data_aug=False,
                          normalisation=normalisation, patch_shape = patch_shape)
    val_dataset = Brats(test, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    if part != 1:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in sub_train]
    else:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in train]
        
def get_datasets_train_rf_withvalid(seed, on="train", fold_number=0, normalisation="minmax", part = 1, all_data = True, patch_shape = 128):
    data_root = get_brats_folder()

    data_file_path = "dataset/train3.txt"
    with open(data_file_path, 'r') as f:
        train_list = [i.strip() for i in f.readlines()]
    train_list.sort()
    
    data_file_path = "dataset/val3.txt"
    with open(data_file_path, 'r') as f:
        test_list = [i.strip() for i in f.readlines()]
    test_list.sort()
    
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    train = []
    test = []
    for l in patients_dir:
        for l2 in train_list:
            if l2 in l:
                train.append(l)
                break
    
    for l in patients_dir:
        for l2 in test_list:
            if l2 in l:
                test.append(l)
                break
                
    train = [Path(l) for l in train]
    test = [Path(l) for l in test]
    
    train_f = open("train.txt", "w")
    csv_writer = csv.writer(train_f)
    
    print("train length: " , len(train))
    print("val length: " , len(test))
    
    if part != 1:
        sub_train = train[:int(len(train) * part)+1]
        if not all_data:
            train = sub_train

    # return patients_dir

    train_dataset = Brats(train, training=True,
                          normalisation=normalisation, patch_shape = patch_shape)
    train_dataset2 = Brats(train, training=False, data_aug=False,
                          normalisation=normalisation, patch_shape = patch_shape)   
    val_dataset = Brats(test, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    if part != 1:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in sub_train]
    else:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in train]

def get_datasets_train_rf_withtest(seed, on="train", fold_number=0, normalisation="minmax", part = 1, all_data = True, patch_shape = 128):
    data_root = get_brats_folder()

    
    data_file_path = "dataset/train3.txt"
    #train_list = []
    with open(data_file_path, 'r') as f:
        train_list = [i.strip() for i in f.readlines()]
    train_list.sort()
    
    data_file_path = "dataset/test3.txt"
    with open(data_file_path, 'r') as f:
        test_list = [i.strip() for i in f.readlines()]
    test_list.sort()
    
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    train = []
    test = []
    for l in patients_dir:
        for l2 in train_list:
            if l2 in l:
                train.append(l)
                break
    
    for l in patients_dir:
        for l2 in test_list:
            if l2 in l:
                test.append(l)
                break
                
    
    train = [Path(l) for l in train]
    #val = [Path(l) for l in val]
    test = [Path(l) for l in test]
    
    train_f = open("train.txt", "w")
    csv_writer = csv.writer(train_f)

    
    print("train length: " , len(train))
    print("val length: " , len(test))
    
   
    
    if part != 1:
        #random.seed(seed)
        #random.shuffle(train)
        sub_train = train[:int(len(train) * part)+1]
        if not all_data:
            train = sub_train

    # return patients_dir

    train_dataset = Brats(train, training=True,
                          normalisation=normalisation, patch_shape = patch_shape)
    train_dataset2 = Brats(train, training=False, data_aug=False,
                          normalisation=normalisation, patch_shape = patch_shape)
    val_dataset = Brats(test, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    if part != 1:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in sub_train]
    else:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in train]

def get_datasets_brats20_rf(seed, on="train", fold_number=0, normalisation="minmax", part = 1, all_data = False, patch_shape = 128):
    data_root = get_brats_folder_20()
    
    data_file_path = data_root + "/train.txt"
    #train_list = []
    with open(data_file_path, 'r') as f:
        train_list = [i.strip()[3:] for i in f.readlines()]
    train_list.sort()
    
    data_file_path = data_root + "val.txt"
    #train_list = []
    with open(data_file_path, 'r') as f:
        train_list.extend([i.strip()[3:] for i in f.readlines()])
    train_list.sort()

    data_file_path = data_root + "test.txt"
    with open(data_file_path, 'r') as f:
        test_list = [i.strip()[3:] for i in f.readlines()]
    test_list.sort()
    
    patients_dir = glob.glob(os.path.join(data_root, "*"))
    print(len(patients_dir))
    patients_dir.sort()
    train = []
    test = []
    for l in patients_dir:
        for l2 in train_list:
            #print(l, l2)
            if l2 in l:
                train.append(l)
                break
    
    for l in patients_dir:
        for l2 in test_list:
            if l2 in l:
                test.append(l)
                break
                
    
    train = [Path(l) for l in train]
    #val = [Path(l) for l in val]
    test = [Path(l) for l in test]
    

    
    print("train length: " , len(train))
    print("val length: " , len(test))
    
   
    
    if part != 1:
        #random.seed(seed)
        #random.shuffle(train)
        sub_train = train[:int(len(train) * part)+1]
        if not all_data:
            train = sub_train

    # return patients_dir

    train_dataset = Brats(train, training=True,
                          normalisation=normalisation, patch_shape = patch_shape)
    train_dataset2 = Brats(train, training=False, data_aug=False,
                          normalisation=normalisation, patch_shape = patch_shape)

    
    val_dataset = Brats(test, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    if part != 1:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in sub_train]
    else:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in train]

#Our implementation for BraTS23
class BratsEval(Dataset):
    def __init__(self, patients_dir, mask_modals=None, benchmarking=False, training=True, data_aug=False,
                 no_seg=False, normalisation="minmax", transforms = None, patch_shape = 128):
        super(BratsEval, self).__init__()
        self.patch_shape = patch_shape
        self.benchmarking = benchmarking
        self.normalisation = normalisation
        self.data_aug = data_aug
        self.training = training
        self.datas = []
        self.validation = no_seg
        self.transforms = eval(transforms or 'Identity()')
        self.patterns = ["-t1n", "-t1c", "-t2w", "-t2f"]
        self.mask_modals = mask_modals
        if not no_seg:
            self.patterns += ["-seg"]
        for patient_dir, mask_modal in zip(patients_dir, mask_modals):
            patient_id = patient_dir.name
            paths = [patient_dir / f"{patient_id}{value}.nii.gz" for value in self.patterns]
            patient = dict(
                id=patient_id, t1=paths[0], t1ce=paths[1],
                t2=paths[2], flair=paths[3], seg=paths[4] if not no_seg else None,
                mask_modal = mask_modal
            )
            self.datas.append(patient)

    def __getitem__(self, idx):
        _patient = self.datas[idx]
        patient_image = {key: self.load_nii(_patient[key]) for key in _patient if key not in ["id", "seg", "mask_modal"]}
        if _patient["seg"] is not None:
            patient_label = self.load_nii(_patient["seg"])
        if self.normalisation == "minmax":
            patient_image = {key: irm_min_max_preprocess(patient_image[key]) for key in patient_image}
        elif self.normalisation == "zscore":
            patient_image = {key: zscore_normalise(patient_image[key]) for key in patient_image}
        patient_image = np.stack([patient_image[key] for key in patient_image])
        
        if patient_image.shape != (4, 155, 240, 240):
            print(f"Different shape: {patient_image.shape}")
        
        if _patient["seg"] is not None:
            et = patient_label == 3
            et_present = 1 if np.sum(et) >= 1 else 0
            tc = np.logical_or(patient_label == 3, patient_label == 1)
            wt = np.logical_or(tc, patient_label == 2)
            patient_label = np.stack([et, tc, wt])
        else:
            patient_label = np.zeros(patient_image.shape)  # placeholders, not gonna use it
            et_present = 0
        if self.training:
            # Remove maximum extent of the zero-background to make future crop more useful
            # Indices of non-zero elements
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Calculates the minimum and maximum indices for each dimension (z, y, x) and Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
    
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
            # default to 128, 128, 128 64, 64, 64 32, 32, 32
            patient_image, patient_label, idx2 = pad_or_crop_image(patient_image, patient_label, target_size=(self.patch_shape, self.patch_shape, self.patch_shape), indices=(zmin, ymin, xmin))
            # Updates the bounding box 
            zmin += idx2[0][0]
            ymin += idx2[1][0]
            xmin += idx2[2][0]

            zmax = zmin + self.patch_shape
            ymax = ymin + self.patch_shape
            xmax = xmin + self.patch_shape

            #if zmin<0 or zmax>155:
            #   print(f"Wrong coordinates for z:{(zmin, zmax)}")

            #print(f"Bounding box after threshold: ({zmin}, {zmax}), ({ymin}, {ymax}), ({xmin}, {xmax})")       
        else:
            z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(patient_image, axis=0) != 0)
            # Add 1 pixel in each side
            zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
            zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
            patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]
        if not self.training:
            patient_image, patient_label = patient_image.astype("float16"), patient_label.astype("bool")
            patient_image, patient_label = [torch.from_numpy(x) for x in [patient_image, patient_label]]
        return dict(patient_id=_patient["id"],
                    image=patient_image, label=patient_label, mask_modal =_patient["mask_modal"],
                    seg_path=str(_patient["seg"]) if not self.validation else str(_patient["t1"]),
                    crop_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
                    et_present=et_present,
                    supervised=True,
                    idx=idx,
                    )
    
    @staticmethod
    def load_nii(path_folder):
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path_folder)))

    def __len__(self):
        return len(self.datas)
    
def get_datasets23_train_rf_forpretrain(seed, on="train", fold_number=0, normalisation="minmax", part = 1, all_data = True, patch_shape = 128):

    # use training set & val set for pretrain.
    data_root = get_brats_folder_23()
    
    data_file_path = ROOT + "dataset/datalist/train.txt"
    #train_list = []
    with open(os.path.join(ROOT, data_file_path), 'r') as f:
        train_list = [i.strip() for i in f.readlines()]
    train_list.sort()

    #data_file_path = "dataset/datalist/val.txt"
    #train_list = []
    #with open(os.path.join(ROOT, data_file_path), 'r') as f:
        #train_list.extend([i.strip() for i in f.readlines()])
    #train_list.sort()
    
    data_file_path = ROOT + "dataset/datalist/test.txt"
    with open(os.path.join(ROOT, data_file_path), 'r') as f:
        test_list = [i.strip() for i in f.readlines()]
    test_list.sort()
    
    patients_dir = glob.glob(os.path.join(data_root, "*"))
    patients_dir.sort()
    train = [] #875
    test = [] #251

    for l in patients_dir:
        for l2 in train_list:
            if l2 in l:
                train.append(l)
                break
    
    for l in patients_dir:
        for l2 in test_list:
            if l2 in l:
                test.append(l)
                break
                
    
    train = [Path(l) for l in train]
    test = [Path(l) for l in test]
    
    train_f = open("train.txt", "w")
    csv_writer = csv.writer(train_f)

    
    print("train length: " , len(train))
    print("val length: " , len(test))
    
   
    if part != 1:
        sub_train = train[:int(len(train) * part)+1]
        if not all_data:
            train = sub_train

    # return patients_dir

    train_dataset = Brats(train, training=True,
                          normalisation=normalisation, patch_shape = patch_shape)
    train_dataset2 = Brats(train, training=False, data_aug=False,
                          normalisation=normalisation, patch_shape = patch_shape)    
    val_dataset = Brats(test, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    if part != 1:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in sub_train]
    else:
        return train_dataset, val_dataset, train_dataset2, [l.name for l in train]
    
def get_datasets23_train_rf_withvalidtest(seed, on="train", fold_number=0, normalisation="minmax", part = 1, all_data = True, patch_shape = 128):
    data_root = get_brats_folder_23()

    data_file_path = ROOT + "dataset/datalist/train.txt"
    with open(data_file_path, 'r') as f:
        train_list = [i.strip() for i in f.readlines()]
    train_list.sort()
    
    data_file_path = ROOT + "dataset/datalist/val15splits_reordered.csv"
    df = pd.read_csv(data_file_path)
    df['mask'] = df['mask'].apply(ast.literal_eval) # modalities we have
    val_mask_modals = df['mask'].apply(lambda x: [i for i, val in enumerate(x) if not val])
    val_list = df['case']
    

    data_file_path = ROOT + "dataset/datalist/test15splits_reordered.csv"
    df = pd.read_csv(data_file_path)
    df['mask'] = df['mask'].apply(ast.literal_eval) # modalities we have
    test_mask_modals = df['mask'].apply(lambda x: [i for i, val in enumerate(x) if not val])
    test_list= df['case']
    
    patients_dir = glob.glob(os.path.join(data_root, "*"))
    patients_dir.sort()
    train = []
    val = []
    test = []
    for l in patients_dir:
        for l2 in train_list:
            if l2 in l:
                train.append(l)
                break
    
    for l in patients_dir:
        for l2 in val_list:
            if l2 in l:
                val.append(l)
                break

    for l in patients_dir:
        for l2 in test_list:
            if l2 in l:
                test.append(l)
                break
                
    train = [Path(l) for l in train]
    val = [Path(l) for l in val]
    test = [Path(l) for l in test]
    
    print("train length: " , len(train))
    print("test length: " , len(test))
    print("val length: " , len(val))
    
    if part != 1:
        sub_train = train[:int(len(train) * part)+1]
        if not all_data:
            train = sub_train

    train_dataset = Brats(train, training=True,
                          normalisation=normalisation, patch_shape = patch_shape)
    train_dataset2 = Brats(train, training=False, data_aug=False,
                          normalisation=normalisation, patch_shape = patch_shape)   
    val_dataset = BratsEval(val, mask_modals=val_mask_modals, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    test_dataset = BratsEval(test, mask_modals=test_mask_modals, training=False, data_aug=False,
                        normalisation=normalisation, patch_shape = patch_shape)
    
    if part != 1:
        return train_dataset, val_dataset, test_dataset, train_dataset2, [l.name for l in sub_train]
    else:
        return train_dataset, val_dataset, test_dataset, train_dataset2, [l.name for l in train]

