from torch.nn import Module, ModuleList
from .conv_block import ConvBlock
from .spdc import SPDC

class FocusGenerator(Module):
    def __init__(self, in_channels, hidden_channels=[]):
        # out_channels = 1
        super().__init__()

        self.spdc_list = ModuleList(SPDC())

    def forward(self, x):

        return x