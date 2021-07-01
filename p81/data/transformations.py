#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 07:54:02 2021

@author: rampfire
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transform_list(mean,std_dev):
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std_dev),
            # A.Rotate(limit = (-7,7),always_apply = True),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=(-7,7), p=.75),
            A.CoarseDropout (max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=mean,p = 0.75),
            # A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[1,1,1], always_apply=True, p=0.5),
            A.HorizontalFlip(p=0.75),
            ToTensorV2()
        ]
    )

def test_transform_list(mean,std_dev):
    return A.Compose(
        [
            A.Normalize(mean=mean, std=std_dev),
            ToTensorV2()
        ]
    )