#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:49:24 2021

@author: rampfire
"""


from .data import train_dataloader_obj,test_dataloader_obj
from .models import custom_resnet
from .training import training,testing
from .logs import logger
from .utils import get_misclassified_images

from torch.optim.lr_scheduler import OneCycleLR,ReduceLROnPlateau
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary
from torch_lr_finder import LRFinder

def lrfinder(model : "model_obj",criterion : "loss_function" = nn.CrossEntropyLoss(),
                optimizer : "optim" = optim.Adam ,lr : float = 0.1,device = "cuda",
                trainloader = None,val_loader = None,end_lr = 1,num_iter = None):
    """
    https://pypi.org/project/torch-lr-finder/
    """
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=1e-2)
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, val_loader=val_loader, end_lr=end_lr, num_iter=num_iter, step_mode="linear")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()
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
    
    net = custom_resnet.Custom_Resnet()
#    print(net)
    model = net.to(config_json["device"])
    metric_log.add_torch_summary = summary(model, input_size=(3, 32, 32))
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    metric_log.add_graph = (model, images,config_json["device"])
    
    lambda_l1 = 0
    lambda_l2 = 0
    
    optimizer = optim.SGD(model.parameters(), lr=config_json["learning_rate"], momentum=0.9, weight_decay = lambda_l2)
#    scheduler = StepLR(optimizer, step_size=70, gamma=0.15)
#    scheduler = ReduceLROnPlateau(optimizer)
    
    lrfinder(model,nn.CrossEntropyLoss(),
                optim.SGD ,lr = 0.0001,device = "cuda",
                trainloader = train_loader,val_loader = test_loader,end_lr = 1,num_iter = 98)
    
    
    
    scheduler = OneCycleLR(optimizer,0.001,98,config_json["epochs"],5)
#    train_loss = []
#    test_loss = []
#    
#    train_accuracy = []
#    test_accuracy = []
    
    for epoch in range(1, config_json["epochs"]):

        tr_loss,tr_acc = training.train(model, config_json["device"], train_loader, nn.CrossEntropyLoss(), optimizer, epoch, lambda_l1)
        tst_loss,tst_acc = testing.test(model, config_json["device"], test_loader, epoch, nn.CrossEntropyLoss(), lambda_l1)
        
        scheduler.step(tst_loss)
        
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







