#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 22:19:56 2021

@author: rampfire
"""
import torch
from .cifar10 import cifar10
from .transformations import train_transform_list

def train_data(train_transforms_list : "Transform_object",
               dataset : str,batch_size : int,shuffle : bool,
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
    train_loader = torch.utils.data.DataLoader(train_dataset(train_transform = train_transforms_list,dataset = dataset),
    batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
    return train_loader

def test_data(test_transforms_list : "Transform_object",
               dataset : str,batch_size : int,shuffle : bool,
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
    test_loader = torch.utils.data.DataLoader(test_dataset(test_transform = test_transforms_list,dataset = dataset),
    batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
    return test_loader


def train_dataset(train_transform, dataset : str, 
                  root_dir : str  = '../data')->"dataset_obj":
    """
    Function which returns the dataset object according to the dataset
    parameter.
    
    Params:
        train_transform : Transform's compose object.
        dataset : MNIST/CIFAR10.
        root_dir : path to store data.
    """
    if(type(dataset) != str):
        raise(TypeError("dataset should be a string"))
    if dataset == "CIFAR10":
        tr_dataset = cifar10(root_dir, train=True,
                                        download=True)
        tr_dataset.tranforms = train_transform_list(tr_dataset.mean,tr_dataset.std_dev)
    else:
        raise(ValueError("Refer doc string for available Datasets"))
    return tr_dataset

def test_dataset(test_transform, dataset : str, 
                  root_dir : str  = '../data')->"dataset_obj":
    """
    Function which returns the dataset object according to the dataset
    parameter.
    
    Params:
        test_transform : Transform's compose object.
        dataset : MNIST/CIFAR10.
        root_dir : path to store data.
    """
    if(type(dataset) != str):
        raise(TypeError("dataset should be a string"))
    if dataset == "CIFAR10":
        ts_dataset = cifar10(root_dir, train=False,
                                        download=True)
    else:
        raise(ValueError("Refer doc string for available Datasets"))
    return ts_dataset