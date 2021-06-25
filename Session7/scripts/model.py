import torch.nn as nn
import torch.nn.functional as F


def base_block(in_channel_size,out_channels,norm_type = "BN",dropout_value = 0.01):
    if type(out_channels) == list and len(out_channels) == 4:
        out_channel_info = out_channels
    elif type(out_channels) == int:
        out_channel_info = [out_channels] * 4
    else:
        raise(TypeError,"out_channels should be a list of len 4 or a int")
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel_size, out_channels=out_channel_info[0], kernel_size=3,padding=1),
        nn.ReLU(),
        normalization_technique(norm_type,out_channel_info[0]),
        nn.Dropout(dropout_value),

        nn.Conv2d(in_channels=out_channel_info[0], out_channels=in_channel_size, kernel_size=3,padding=1, dilation = 1),
        nn.ReLU(),
        normalization_technique(norm_type,in_channel_size),
        nn.Dropout(dropout_value),
        
        # nn.Conv2d(in_channels=in_channel_size, out_channels=out_channel_info[0], kernel_size=3,padding=1),
        # nn.ReLU(),
        # normalization_technique(norm_type,out_channel_info[0]),
        # nn.Dropout(dropout_value),

        # nn.Conv2d(in_channels=out_channel_info[0], out_channels=out_channel_info[0], kernel_size=1),
        # nn.ReLU(),
        # normalization_technique(norm_type,out_channel_info[0]),
        # nn.Dropout(dropout_value)
        nn.Conv2d(
            in_channels=in_channel_size,
            out_channels=in_channel_size,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel_size
        ),
        nn.ReLU(),
        normalization_technique(norm_type,in_channel_size),
        nn.Dropout(dropout_value),
        nn.Conv2d(
            in_channels=in_channel_size,
            out_channels=out_channel_info[0],
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        ),
        nn.ReLU(),
        normalization_technique(norm_type,out_channel_info[0]),
        nn.Dropout(dropout_value)

    )

def normalization_technique(normalization,in_channels):
    if type(normalization) not in [int,list] and type(in_channels) != int:
        raise(TypeError,f"normalization should be a int or list,in_channels should be int")

    if normalization == "GN":
        return nn.GroupNorm(2,in_channels)
    elif normalization == "LN":
        return nn.GroupNorm(1,in_channels)
    elif normalization == "BN":
        return nn.BatchNorm2d(in_channels)
    else:
        raise(ValueError,"normalization should be GN or LN or BN")

class Net(nn.Module):
    def __init__(self ,norm_type : "BN/LN/GN"):
        super(Net, self).__init__()
        self.convblock1 = base_block(3,36)
        # self.maxpool1 = nn.MaxPool2d(2, 2)
        self.convblock2 = base_block(36,36)
        self.maxpool2 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=36, 
                                  kernel_size=3, dilation = 2,stride = 2),
                                    nn.ReLU(),
                                    normalization_technique("BN",36),
                                    nn.Dropout(0.01))
        self.convblock3 = base_block(36,60)
        # self.maxpool3 = nn.MaxPool2d(2, 2)
        self.convblock4 = base_block(60,65)
        self.maxpool4 = nn.Sequential(nn.Conv2d(in_channels=65, out_channels=70, 
                                  kernel_size=3, dilation = 2,stride = 2),
                                    nn.ReLU(),
                                    normalization_technique("BN",70),
                                    nn.Dropout(0.01))

        self.output_layer = nn.Conv2d(in_channels=70, out_channels=10, kernel_size=1)
        self.gap = nn.Sequential(
                nn.AdaptiveAvgPool2d(1)
            )
    def forward(self, x):
        x = self.convblock1(x)
        # x = self.maxpool1(x)

        x = self.convblock2(x)
        x = self.maxpool2(x)

        x = self.convblock3(x)
        # x = self.maxpool3(x)

        x = self.convblock4(x)
        x = self.maxpool4(x)

        x = self.output_layer(x)

        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)