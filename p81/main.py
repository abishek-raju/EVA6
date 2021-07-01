#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:49:24 2021

@author: rampfire
"""


from .data import train_dataloader_obj,test_dataloader_obj
from .models import resnet
from .training import training,testing

from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F


def main(config_json):
    train_loader = train_dataloader_obj(dataset = config_json["dataset"],
                                      batch_size = config_json["tr_batch_size"],
                                      dataloader_kwargs = config_json["dev_kwargs"])
    
    test_loader = test_dataloader_obj(dataset = config_json["dataset"],
                                      batch_size = config_json["tst_batch_size"],
                                      dataloader_kwargs = config_json["dev_kwargs"])
    
    net = resnet.ResNet18()
#    print(net)
    model = net.to(config_json["device"])
    
    lambda_l1 = 0
    lambda_l2 = 0
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay = lambda_l2)
    scheduler = StepLR(optimizer, step_size=70, gamma=0.15)
    train_loss = []
    test_loss = []
    
    train_accuracy = []
    test_accuracy = []
    
    for epoch in range(1, 100):

        tr_loss,tr_acc = training.train(model, config_json["device"], train_loader, F.nll_loss, optimizer, epoch, lambda_l1)
        scheduler.step()
        tst_loss,tst_acc = testing.test(model, config_json["device"], test_loader, epoch, F.nll_loss, lambda_l1)
        train_loss.append(tr_loss),train_accuracy.append(tr_acc)
        test_loss.append(tst_loss),test_accuracy.append(tst_acc)
        print("Train_epoch : ",100*tr_acc.item())
        print("Test_epoch : ",100*tst_acc.item())
        print("Learning Rate : ",optimizer.param_groups[0]['lr'])
#    writer.add_scalars('Loss',
#    {
#    'train_loss': tr_loss,
#    'test_loss': tst_loss,
#    },epoch)
#    writer.add_scalars('Accuracy',
#    {
#    'train_accuracy': tr_acc,
#    'test_accuracy': tst_acc,
#    },epoch)
#    writer.flush()













