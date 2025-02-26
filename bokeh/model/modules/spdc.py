# from torch import nn
from torch import mean, stack
from torch.nn import Module, Linear, Hardtanh, Tanh
from .conv_block import ConvBlock

class SPDC(Module):
    def __init__(self, in_channels, hidden_channels, distance=[1, 3, 5], alpha=0.5, has_skip_connection=True):
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

        self.weight_mlp = Linear(3, 3, bias=False)
        self.act = Tanh()

        self.alpha = alpha
        self.has_skip_connection = has_skip_connection

    def forward(self, x):
        feature = self.pw_expand(x)

        out_0 = self.dw_0(feature)
        out_1 = self.dw_1(out_0)
        out_2 = self.dw_2(out_1)

        if self.alpha <= 0.0:
            out = out_0 + out_1 + out_2
        else:
            weight_0 = mean(out_0, [2, 3], keepdim=False)
            weight_1 = mean(out_1, [2, 3], keepdim=False)
            weight_2 = mean(out_2, [2, 3], keepdim=False)
            weight = stack((weight_0, weight_1, weight_2), dim=2)
            weight = weight + self.act(self.weight_mlp(weight)) * self.alpha
            weight = weight.unsqueeze(3)
            # print(weight.shape)
            out = out_0 * weight[:, :, 0:1, :] + out_1 * weight[:, :, 1:2, :] + out_2[:, :, 2:3, :]

        out = self.pw_project(out)

        if self.has_skip_connection:
            out = out + x

        return out