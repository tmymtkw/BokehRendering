from torch.nn import Module
from torch import sum

class LossHolder(Module):
    def __init__(self):
        self.functions = []

    def forward(self, x):
        losses = [func(x) * scale for (func, scale) in self.functions]
        return sum()