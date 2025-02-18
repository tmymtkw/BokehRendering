from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 3, 3, 1, 1),
                               nn.ReLU())

    def forward(self, x):
        x = self.f(x)

        return x