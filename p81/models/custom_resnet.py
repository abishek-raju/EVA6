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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1,self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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
        self.r1 = self._make_layer(BasicBlock, 128, 256, stride=2)
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
        
        self.linear_out = nn.Sequential(
            nn.Linear(512, 10),
            nn.ReLU()
            )


    def _make_layer(self, block,in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion
        return nn.Sequential(*layers)



    def forward(self, x):
        x = self.preplayer(x)

        x = self.layer_1(x)
        x = self.r1(x) + x

        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.r2(x) + x

        x = self.pool(x)

        x = x.view(-1,512)
        
        x = self.linear_out(x)
        x = F.softmax(x,dim=-1)
        return x