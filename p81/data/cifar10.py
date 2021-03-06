#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:53:48 2021

@author: rampfire
"""


from torchvision import datasets
import numpy as np


class cifar10:
    """
    cifar10 dataset class which call the transformations to augment the data
    """
    def __init__(self, root_dir : str  = '../data',
                 train : bool = False , download : bool = True)->"dataset_obj":
        self.cifar_ = datasets.CIFAR10(root_dir, train=train,
                                        download=download)
        self._mean = None
        self._std_dev = None
        self._transform = None

    def __getitem__(self,index):
        image, label = self.cifar_[index]
        if self._transform:
            return self._transform(image = np.asarray(image))["image"],label
        else:
            return image,label
    def __len__(self):
        return len(self.cifar_.data)
    
    @property
    def mean(self):
        return self.cifar_.data.mean(axis=(0,1,2))/255
    
    @property
    def std_dev(self):
        return self.cifar_.data.std(axis=(0,1,2))/255
    
    @property
    def transforms(self):
        return self._transform
    @transforms.setter
    def transforms(self,transforms):
        self._transform = transforms
    
    
    