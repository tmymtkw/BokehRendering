from torch.nn import Module, Conv2d, BatchNorm2d, ReLU

class ConvBlock(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=False):
        super().__init__()
        
        self.conv = Conv2d(in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=bias)
        
        self.norm = BatchNorm2d(num_features=out_channels)

        self.relu = ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))