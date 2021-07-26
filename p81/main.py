#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:49:24 2021

@author: rampfire
"""


from .data import train_dataloader_obj,test_dataloader_obj
from .models import resnet
from .training import training,testing
from .logs import logger
from .utils import get_misclassified_images

from torch.optim.lr_scheduler import OneCycleLR,ReduceLROnPlateau,StepLR
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
#from torch_lr_finder import LRFinder

def lrfinder(model : "model_obj",criterion : "loss_function" = nn.CrossEntropyLoss(),
                optimizer : "optim" = optim.Adam ,lr : float = 0.1,device = "cuda",
                trainloader = None,val_loader = None,end_lr = 1,num_iter = None):
    """
    https://pypi.org/project/torch-lr-finder/
    """
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
    lr_finder.plot()
#    lr_finder.reset()
#    lr_finder.unfreeze()
#    lr_finder.lr_find()
#    lr_finder.recorder.plot()
    
    
def main(config_json):
    metric_log = logger.log_training_params()
    metric_log.add_text = config_json
    train_loader = train_dataloader_obj(dataset = config_json["dataset"],
                                      batch_size = config_json["tr_batch_size"],
                                      dataloader_kwargs = config_json["dev_kwargs"])
    
    test_loader = test_dataloader_obj(dataset = config_json["dataset"],
                                      batch_size = config_json["tst_batch_size"],
                                      dataloader_kwargs = config_json["dev_kwargs"])
    
    net = resnet.ResNet18()
#    print(net)
    model = net.to(config_json["device"])
    print("here2")
    metric_log.add_torch_summary = summary(model, input_size=(3, 64, 64))
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
#    print(images,labels)
    metric_log.add_graph = (model, images.float(),config_json["device"])
    
    lambda_l1 = 0
    lambda_l2 = 0
    print("here1")
    

    
#    lrfinder(model,nn.CrossEntropyLoss(),
#                optim.SGD ,lr = 1e-7,device = "cuda",
#                trainloader = train_loader,val_loader = test_loader,end_lr = 100,num_iter = 98)
#    
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = lambda_l2)
#    optimizer = optim.Adam(model.parameters(), lr=0.5)
#    scheduler = OneCycleLR(optimizer, max_lr=0.008, steps_per_epoch=98,
#                                                  epochs=24,
#                                                  pct_start=5/24, 
#                                                  anneal_strategy='linear')
#    scheduler = StepLR(optimizer, step_size=5, gamma=0.15)
#    scheduler = ReduceLROnPlateau(optimizer)
    scheduler = OneCycleLR(optimizer, 
                            max_lr=1.0,
                            steps_per_epoch=len(train_loader), 
                            epochs=config_json["epochs"],
                            pct_start=0.5,
                            div_factor=10,
                            three_phase=True,
                            anneal_strategy='linear'
                            )
#    train_loss = []
#    test_loss = []
#    
#    train_accuracy = []
#    test_accuracy = []
    print("run here")
    
    for epoch in range(1, config_json["epochs"]):

        tr_loss,tr_acc = training.train(model, config_json["device"], train_loader, nn.CrossEntropyLoss(), optimizer, scheduler,epoch, lambda_l1)
        print("run here")
        tst_loss,tst_acc = testing.test(model, config_json["device"], test_loader, epoch, nn.CrossEntropyLoss(), lambda_l1)
        
#        scheduler.step(tr_loss)
#        optimizer.step()
#        scheduler.step()
        
#        train_loss.append(tr_loss),train_accuracy.append(tr_acc)
#        test_loss.append(tst_loss),test_accuracy.append(tst_acc)
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

        metric_log.train_test_loss = tr_loss,tst_loss,epoch
        metric_log.train_test_accuracy = tr_acc,tst_acc,epoch
        metric_log.flush()
    
    metric_log.misclassified_images = get_misclassified_images.get_misclassified_images(config_json["max_misclassified_images"],
                                                      test_loader,
                                                      train_loader.dataset.cifar_.classes,
                                                      config_json["device"],
                                                      model)
    metric_log.classified_images = get_misclassified_images.get_classified_images(config_json["max_classified_images"],
                                                  test_loader,
                                                  train_loader.dataset.cifar_.classes,
                                                  config_json["device"],
                                                  model)

    return model,metric_log







