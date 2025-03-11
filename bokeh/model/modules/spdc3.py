# from torch import nn
from torch import mean, stack
from torch.nn import Module, Linear, Sequential, ReLU, Sigmoid
from .conv_block import ConvBlock

class SPDC3(Module):
    def __init__(self, in_channels, hidden_channels, distance=[1, 3, 5], reduction=4, has_skip_connection=True):
        super().__init__()
        self.pw_expand = ConvBlock(in_channels, hidden_channels, kernel_size=1)

        self.dw_0 = ConvBlock(hidden_channels, hidden_channels, kernel_size=3,
                              stride=1, padding=distance[0], dilation=distance[0],
                              groups=hidden_channels, bias=False)
        
        self.dw_1 = ConvBlock(hidden_channels, hidden_channels, kernel_size=3,
                              stride=1, padding=distance[1], dilation=distance[1],
                              groups=hidden_channels, bias=False)
        
        self.dw_2 = ConvBlock(hidden_channels, hidden_channels, kernel_size=3,
                              stride=1, padding=distance[2], dilation=distance[2],
                              groups=hidden_channels, bias=False)

        self.pw_project = ConvBlock(hidden_channels, in_channels, kernel_size=1)

        self.weight_mlp = Sequential(Linear(3, 12, bias=False),
                                     ReLU(),
                                     Linear(12, 3, bias=False),
                                     Sigmoid())

        self.se_layer = Sequential(Linear(hidden_channels, hidden_channels//reduction, bias=False),
                                   ReLU(),
                                   Linear(hidden_channels//reduction, hidden_channels, bias=False),
                                   Sigmoid())
        
        self.has_skip_connection = has_skip_connection

    def forward(self, x):
        feature = self.pw_expand(x)

        out_0 = self.dw_0(feature)
        out_1 = self.dw_1(out_0)
        out_2 = self.dw_2(out_1)

        # calculate channel attention weight
        weight = stack((mean(out_0, [2, 3], keepdim=False), 
                        mean(out_1, [2, 3], keepdim=False), 
                        mean(out_2, [2, 3], keepdim=False)), dim=2)
        weight_t = weight.transpose(1, 2)
        weight = self.se_layer(weight_t).transpose(1, 2) + self.weight_mlp(weight)
        weight = weight.unsqueeze(3)

        out = (out_0 * weight[:, :, 0:1, :]
               + out_1 * weight[:, :, 1:2, :]
               + out_2 * weight[:, :, 2:3, :]) * 0.5

        out = self.pw_project(out)

        if self.has_skip_connection:
            out = out + x

        return out