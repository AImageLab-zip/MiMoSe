import numpy as np
import torch


def mask_gen(Batchsize, NumHead, patches, NumClass):
    attn_shape = (patches*NumClass, patches*NumClass)
    self_mask = np.zeros(attn_shape)
    for i in range(NumClass):
        self_mask[patches*i:patches*(i+1),patches*i:patches*(i+1)] = 1
    self_mask = torch.from_numpy(self_mask)
    self_mask = torch.unsqueeze(self_mask, 0).repeat(NumHead,1,1)
    self_mask = torch.unsqueeze(self_mask, 0).repeat(Batchsize, 1, 1, 1)


    return self_mask == 1

def mask_gen_fusion(Batchsize, NumHead, patches, NumClass, mask):
    attn_shape = (patches*(NumClass+1), patches*(NumClass+1))
    self_mask = np.zeros(attn_shape)
    for i in range(NumClass):
        self_mask[patches*i:patches*(i+1),patches*i:patches*(i+1)] = 1
    self_mask[patches*NumClass:patches*(NumClass+1),:] = 1
    for i in range(NumClass):
        if mask[0][i] == 0:
            self_mask[patches*NumClass:patches*(NumClass+1),patches*i:patches*(i+1)] = 0
    self_mask = torch.from_numpy(self_mask)
    self_mask = torch.unsqueeze(self_mask, 0).repeat(NumHead,1,1)
    self_mask = torch.unsqueeze(self_mask, 0).repeat(Batchsize, 1, 1, 1)


    return self_mask == 1

def mask_gen_skip(Batchsize, NumHead, patches, NumClass, mask):
    attn_shape = (patches*(NumClass+1), patches*(NumClass+1))
    self_mask = np.zeros(attn_shape)
    for i in range(NumClass):
        self_mask[patches*i:patches*(i+1),patches*i:patches*(i+1)] = 1
    self_mask[patches*NumClass:patches*(NumClass+1),:] = 1
    for i in range(NumClass):
        if mask[0][i] == 0:
            self_mask[patches*NumClass:patches*(NumClass+1),patches*i:patches*(i+1)] = 0
    self_mask = torch.from_numpy(self_mask)
    self_mask = torch.unsqueeze(self_mask, 0).repeat(NumHead,1,1)
    self_mask = torch.unsqueeze(self_mask, 0).repeat(Batchsize, 1, 1, 1)


    return self_mask == 1

def mask_gen_cross4(Batchsize, K, C, mask):
    attn_shape = (K, C)
    self_mask = np.ones(attn_shape)
    for i in range(4):
        if mask[0][i] == 0:
            self_mask[:,(C//4)*i:(C//4)*(i+1)] = 0

    self_mask = torch.from_numpy(self_mask)

    self_mask = torch.unsqueeze(self_mask, 0).repeat(Batchsize, 1, 1)

    return self_mask == 1
