from torch import mean, pow
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img_output, img_target):
        return mean(pow(img_output - img_target, 2)) + 1.0e-6