#coding:utf-8

import os
import torch
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat
import numpy as np
import glob
from torchvision import transforms
import random

import matplotlib.pyplot as plt


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class NYUv2(data.Dataset):
    """NYUv2 depth dataset loader.
    
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **num_classes** (string, optional): The number of classes, must be 40 or 13. Default:13.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A list of function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with labels or not. Default: 'labeled'.
    """
    cmap = colormap()

    def __init__(self,
                 root,
                 split='train',
                 num_classes=13,
                 transform=None,
                 ds_type='labeled'):

        assert(split in ('train', 'test'))
        assert(ds_type in ('labeled', 'unlabeled'))
        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transform = transform
        self.num_classes = num_classes
        self.train_idx = np.array([255, ] + list(range(num_classes)))

        if ds_type == 'labeled':
            split_mat = loadmat(os.path.join(
                self.root, 'nyuv2-meta-data', 'splits.mat'))

            idxs = split_mat[self.split+'Ndxs'].reshape(-1)

            self.images = [os.path.join(self.root, '480_640', 'IMAGE', '%d.png' % (idx-1))
                           for idx in idxs]
            if self.num_classes == 13:
                self.targets = [os.path.join(self.root, 'nyuv2-meta-data', '%s_labels_13' % self.split, 'new_nyu_class13_%04d.png' % idx)
                                for idx in idxs]
            elif self.num_classes == 40:
                self.targets = [os.path.join(self.root, '480_640', 'SEGMENTATION', '%04d.png' % idx)
                                for idx in idxs]
            else:
                raise ValueError(
                    'Invalid number of classes! Please use 13 or 40')
        else:
            self.images = [glob.glob(os.path.join(
                self.root, 'unlabeled_images/*.png'))]
        print(self.split, len(self.images))


    def __getitem__(self, idx):
        if self.ds_type == 'labeled':
            image = Image.open(self.images[idx])
            target = Image.open(self.targets[idx])

            if self.transform:
                image, target = self.transform(image, target)
            #print(target)
            target = self.train_idx[target]
            return image, target
        else:
            image = Image.open(self.images[idx])
            if self.transforms is not None:
                image = self.transforms(image)
            image = transforms.ToTensor()(image)
            return image, None

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        target = (target+1).astype('uint8')  # 255 -> 0, 0->1, 1->2
        return cls.cmap[target]

class NYUv2Depth(data.Dataset):
    """NYUv2 depth dataset loader.
    
    **Parameters:**
        - **root** (string): Root directory path.
        - **split** (string, optional): 'train' for training set, and 'test' for test set. Default: 'train'.
        - **num_classes** (string, optional): The number of classes, must be 40 or 13. Default:13.
        - **transform** (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
        - **target_transforms** (callable, optional): A list of function/transform that takes in the target and transform it. Default: None.
        - **ds_type** (string, optional): To pick samples with labels or not. Default: 'labeled'.
    """
    cmap = colormap()

    def __init__(self,
                 root,
                 split='train',
                 num_classes=13,
                 transform=None,
                 #target_transforms=None,
                 ds_type='labeled'):

        assert(split in ('train', 'test'))
        assert(ds_type in ('labeled', 'unlabeled'))

        self.root = root
        self.split = split
        self.ds_type = ds_type
        self.transform = transform

        self.num_classes = num_classes

        self.train_idx = np.array([255, ] + list(range(num_classes)))
        
        if ds_type == 'labeled':
            split_mat = loadmat(os.path.join(
                self.root, 'nyuv2-meta-data', 'splits.mat'))

            idxs = split_mat[self.split+'Ndxs'].reshape(-1)
            self.images = [os.path.join(self.root, '480_640', 'IMAGE', '%d.png' % (idx-1))
                           for idx in idxs]
            if self.num_classes == 13:
                self.targets = [os.path.join(self.root, 'nyuv2-meta-data', '%s_labels_13' % self.split, 'new_nyu_class13_%04d.png' % idx)
                                for idx in idxs]
            elif self.num_classes == 40:
                self.targets = [os.path.join(self.root, '480_640', 'SEGMENTATION', '%04d.png' % idx)
                                for idx in idxs]
            else:
                raise ValueError(
                    'Invalid number of classes! Please use 13 or 40')
            self.depths = [os.path.join(
                self.root, 'FINAL_480_640', 'DEPTH', '%04d.png' % idx) for idx in idxs]
        else:
            self.images = [glob.glob(os.path.join(
                self.root, 'unlabeled_images/*.png'))]

    def __getitem__(self, idx):
        if self.ds_type == 'labeled':
            image = Image.open(self.images[idx])
            depth = Image.open(self.depths[idx])
            #print(np.array(depth,dtype='float').max())
            if self.transform:
                image, depth = self.transform(image, depth)
            return image, depth / 1000
        else:
            image = Image.open(self.images[idx])
            if self.transform is not None:
                image = self.transform(image)
            #image = transforms.ToTensor()(image)
            return image, None

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, target):
        cm = plt.get_cmap('jet')
        target = (target/7).clip(0,1)
        target = cm(target)[:,:,:,:3]
        return target
        
