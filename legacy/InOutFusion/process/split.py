
import os
import random
import argparse
import numpy as np
from utils import check_dir

def split(brain_dir, save_dir, n_train=260, n_val=40, n_split=3):
    patient_ids = []
    for f in os.listdir(brain_dir):
        patient_ids.append(f)
    patient_ids = np.array(patient_ids)
    # patient_ids = np.array([f for f in os.listdir(brain_dir)])
    perm = np.random.permutation(len(patient_ids))
    patient_ids = patient_ids[perm]

    n_test = len(patient_ids) - n_train - n_val

    splits = []
    i = random.randint(1, n_train)
    for ns in range(n_split):
        splits.append(dict(val=patient_ids[i: i + n_val],
                           test=patient_ids[i+ n_val: i + n_val + n_test],
                           train=np.concatenate([patient_ids[:i], patient_ids[i + n_val + n_test :]])
                           ))
        perm = np.random.permutation(len(patient_ids))
        patient_ids = patient_ids[perm]
        i = random.randint(1, n_train)

    check_dir(save_dir)
    for i in range(n_split):

        train_f = os.path.join(save_dir, '{}-train.txt'.format(i))
        val_f = os.path.join(save_dir, '{}-val.txt'.format(i))
        test_f = os.path.join(save_dir, '{}-test.txt'.format(i))
        train_f = open(train_f, 'w')
        val_f = open(val_f, 'w')
        test_f = open(test_f, 'w')

        writer = dict(train=train_f, val=val_f, test=test_f)
        lines = dict(train=[], val=[], test=[])
        for phase in ('train', 'val', 'test'):
            for pid in sorted(splits[i][phase]):
                target_dir = os.path.join(brain_dir, pid)

                fpath = target_dir

                line = '{}\n'.format(fpath)
                line = line.replace('\\','/')
                line = line.replace('../dataset','./dataset')

                lines[phase].append(line)

        for phase in ('train', 'val', 'test'):
            tar_list = lines[phase]
            for line in tar_list:
                writer[phase].write(line)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--brain_dir', type=str, default='/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/')
    #parse.add_argument('--brain_dir', type=str, default='../dataset/MICCAI_BraTS_2018_Data_Training/HGG')
    parse.add_argument('--save_dir', type=str, default='./partition')
    parse.add_argument('--n_split', type=int, default=3)
    parse.add_argument('--seed', type=int, default=1234)
    opt = parse.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)

    split(opt.brain_dir, opt.save_dir, n_split=opt.n_split)
    print('Done!')
