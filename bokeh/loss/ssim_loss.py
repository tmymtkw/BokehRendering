from torch.nn import Module
from torch.nn.functional import conv2d
from util.functions.gaussian import getKernel

class SSIMLoss(Module):
    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = getKernel(kernel_size, sigma)
        self.padding = kernel_size // 2
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, img_output, img_target):
        kernel = self.kernel.to(device=img_target.device)
        # if self.kernel.device != output.device:
        #     self.kernel = self.kernel.to(output.device)
        return 1 - self.GetSSIM(img_output, img_target, kernel).mean()
        
    def GetSSIM(self, s, t, kernel):
        # Compute means
        ux = conv2d(s, kernel, padding=self.padding, groups=3)
        uy = conv2d(t, kernel, padding=self.padding, groups=3)
        # Compute variances
        uxx = conv2d(s * s, kernel, padding=self.padding, groups=3)
        uyy = conv2d(t * t, kernel, padding=self.padding, groups=3)
        uxy = conv2d(s * t, kernel, padding=self.padding, groups=3)

        # print(ux.shape, uy.shape, uxx.shape, uyy.shape, uxy.shape)
        return ((2 * ux * uy + self.c1) * (2 * (uxy - ux * uy) + self.c2) 
                / (ux ** 2 + uy ** 2 + self.c1) / ((uxx - ux * ux) + (uyy - uy * uy) + self.c2))
            
    def __repr__(self):
        return "SSIM"