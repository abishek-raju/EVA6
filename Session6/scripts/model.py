import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.01)
        )
        #28/26/3
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.01)
        )
        #26/24/5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        #24/22/7

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(0.01)
        )
        #22/20/9
        self.pool = nn.MaxPool2d(2, 2)
        #20/10/10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=14, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.01)
        )
        #10/8/14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.01)
        )
        #8/6/18
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.01)
        )
        #6/6/18
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)