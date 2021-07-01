#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:49:24 2021

@author: rampfire
"""


from .data import train_dataloader_obj,test_dataloader_obj
from .models import resnet





def main(config_json):
    train_data = train_dataloader_obj(dataset = config_json["dataset"],
                                      batch_size = config_json["tr_batch_size"],
                                      dataloader_kwargs = config_json["dev_kwargs"])
    
    test_data = test_dataloader_obj(dataset = config_json["dataset"],
                                      batch_size = config_json["tst_batch_size"],
                                      dataloader_kwargs = config_json["dev_kwargs"])
    
    net = resnet.ResNet18()
    print(net)