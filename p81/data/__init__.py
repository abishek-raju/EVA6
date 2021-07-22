#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 22:19:56 2021

@author: rampfire
"""
import torch
from .cifar10 import cifar10
from ..utils.transformations import train_transform_list,test_transform_list

def train_dataloader_obj(dataset : str,batch_size : int,shuffle : bool = True,
               dataloader_kwargs : dict = None)->"dataloader_obj":
    """
    Function which returns the dataloader object according to the dataset
    parameter.
    
    Params:
        train_transforms_list : Transform's compose object.
        dataset : MNIST/CIFAR10.
        root_dir : path to store data.
        batch_size : batch_size
        shuffle : True/False
        dataloader_kwargs : {'num_workers': 8, 'pin_memory': True}
    """
    train_loader = torch.utils.data.DataLoader(train_dataset(dataset = dataset),
    batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
    return train_loader

def test_dataloader_obj(dataset : str,batch_size : int,shuffle : bool = True,
               dataloader_kwargs : dict = None)->"dataloader_obj":
    """
    Function which returns the dataloader object according to the dataset
    parameter.
    
    Params:
        test_transforms_list : Transform's compose object.
        dataset : MNIST/CIFAR10.
        root_dir : path to store data.
        batch_size : batch_size
        shuffle : True/False
        dataloader_kwargs : {'num_workers': 8, 'pin_memory': True}
    """
    test_loader = torch.utils.data.DataLoader(test_dataset(dataset = dataset),
    batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
    return test_loader


def train_dataset(dataset : str, 
                  root_dir : str  = '../data')->"dataset_obj":
    """
    Function which returns the dataset object according to the dataset
    parameter.
    
    Params:
        train_transform : Transform's compose object.
        dataset : MNIST/CIFAR10/TINY_IMAGENET_200.
        root_dir : path to store data.
    """
    if(type(dataset) != str):
        raise(TypeError("dataset should be a string"))
    if dataset == "CIFAR10":
        tr_dataset = cifar10(root_dir, train=True,
                                        download=True)
        tr_dataset.transforms = train_transform_list(tr_dataset.mean,tr_dataset.std_dev)
    elif dataset == "TINY_IMAGENET_200":
        tr_dataset = Tiny_image_net_Dataset_200()
        tr_dataset.transforms = train_transform_list(mean=[0.485, 0.456, 0.406], std_dev=[0.229, 0.224, 0.225])
    else:
        raise(ValueError("Refer doc string for available Datasets"))
    return tr_dataset

def test_dataset(dataset : str, 
                  root_dir : str  = '../data')->"dataset_obj":
    """
    Function which returns the dataset object according to the dataset
    parameter.
    
    Params:
        test_transform : Transform's compose object.
        dataset : MNIST/CIFAR10/TINY_IMAGENET_200.
        root_dir : path to store data.
    """
    if(type(dataset) != str):
        raise(TypeError("dataset should be a string"))
    if dataset == "CIFAR10":
        ts_dataset = cifar10(root_dir, train=False,
                                        download=True)
        ts_dataset.transforms = test_transform_list(ts_dataset.mean,ts_dataset.std_dev)
    elif dataset == "TINY_IMAGENET_200":
        ts_dataset = Tiny_image_net_Dataset_200(tr_tst = "test")
        ts_dataset.transforms = test_transform_list(mean=[0.485, 0.456, 0.406], std_dev=[0.229, 0.224, 0.225])
    else:
        raise(ValueError("Refer doc string for available Datasets"))
    return ts_dataset