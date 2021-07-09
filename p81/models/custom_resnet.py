import torch.nn as nn
import torch.nn.functional as F

def ResBlock(in_planes, planes, pading=1):
    return nn.Sequential(
    nn.Conv2d(
        in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
    nn.GroupNorm(1,planes),
    nn.Conv2d(planes, planes, kernel_size=3,
                           stride=1, padding=1, bias=False),
    nn.GroupNorm(1,planes)
    )

def block(in_planes, planes, pading=1,max_pool = True):
    return nn.Sequential(
    nn.Conv2d(in_planes, planes, 3, padding=pading, bias=False),
    nn.MaxPool2d(2,2),
    nn.BatchNorm2d(planes),
    nn.ReLU()
    )
class Custom_Resnet(nn.Module):
    def __init__(self):
        super(Custom_Resnet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(64), 
                                   nn.ReLU())
        self.pool1 = block(64,128)
        self.conv2 = ResBlock(128,128)
        self.pool2 = block(128,256)
        self.pool3 = block(256,512)   
        self.conv3 = ResBlock(512,512)
        self.max_pool = nn.MaxPool2d(4,4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x + self.conv2(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = x + self.conv3(x)
        x = self.max_pool(x)
        x = x.view(-1,512)
        x = self.fc(x)
        x = x.view(-1,10)
        return F.log_softmax(x, dim=-1)