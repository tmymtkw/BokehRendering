from torch import mean, pow
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_input, img_target):
        return mean(pow(img_input - img_target, 2))