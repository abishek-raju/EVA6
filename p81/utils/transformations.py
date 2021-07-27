#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 07:54:02 2021

@author: rampfire
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transform_list(mean,std_dev):
#    return A.Compose(
#        [   
#            A.Normalize(mean=mean, std=std_dev),
#            A.Rotate(limit = (-5,5),always_apply = True),
#            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=(-7,7), p=.75),
#            A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=mean,p = 0.75),
#            A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=mean, always_apply=True, p=0.5),
#            A.HorizontalFlip(p=0.75),
#            A.RandomCrop(32,32),
#            ToTensorV2()
#        ]
#    )
    return A.Compose([
      
      A.PadIfNeeded(min_height=76, min_width=76, always_apply=True),
      A.RandomCrop(64,64),
      A.Rotate(limit=15),
      A.CoarseDropout(1,24, 24, 1, 8, 8,fill_value=[m*255 for m in [0.4803, 0.4482, 0.3976]]),
      A.VerticalFlip(),
      A.HorizontalFlip(),
      A.Normalize([0.4803, 0.4482, 0.3976], [0.2766, 0.2691, 0.2819]),
      ToTensorV2()
    ])

def test_transform_list(mean,std_dev):
    return A.Compose(
        [   
            A.Normalize([0.4803, 0.4482, 0.3976], [0.2766, 0.2691, 0.2819]),
            ToTensorV2()

        ]
    )
#import torchvision.transforms as transforms
#def train_transform_list(mean,std_dev):
#    return transforms.Compose([
#        transforms.ToTensor(),
##        transforms.Normalize(mean=mean, std=std_dev)
#    ])
#def test_transform_list(mean,std_dev):
#    return transforms.Compose([
#        transforms.ToTensor(),
##        transforms.Normalize(mean=mean, std=std_dev)
#    ])