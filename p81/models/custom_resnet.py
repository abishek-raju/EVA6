# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

def normalization_technique(normalization,in_channels):
    if normalization == "GN":
        return nn.GroupNorm(2,in_channels)
    elif normalization == "LN":
        return nn.GroupNorm(1,in_channels)
    else:
        return nn.BatchNorm2d(in_channels)


class Custom_Resnet(nn.Module):
    def __init__(self ,norm_type : "BN/LN/GN" = "BN"):
        super(Custom_Resnet, self).__init__()
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding = 1),
            normalization_technique(norm_type,64),
            nn.ReLU(),
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding = 1),
            nn.MaxPool2d(2, 2),
            normalization_technique(norm_type,128),
            nn.ReLU()
        )
        self.r1 = BasicBlock(128,128)
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding = 1),
            nn.MaxPool2d(2, 2),
            normalization_technique(norm_type,256),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding = 1),
            nn.MaxPool2d(2, 2),
            normalization_technique(norm_type,512),
            nn.ReLU()
        )
        self.r2 = BasicBlock(512,512)

        self.pool = nn.MaxPool2d(4,4)
        
        self.linear = nn.Linear(512, 10)

    def forward(self, x):
        x = self.preplayer(x)

        x = self.layer_1(x)
        x = self.r1(x) + x

        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.r2(x) + x

        x = self.pool(x)

        x = x.view(x.size(0), -1)
        
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        return x