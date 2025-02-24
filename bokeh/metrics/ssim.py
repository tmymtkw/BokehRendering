import torch
from torch.nn.functional import conv2d
from torch import Tensor, arange, exp, pow, matmul, float32

class SSIM():
    def __init__(self, kernel_size=11, sigma=1.5):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.GetKernel(kernel_size, sigma)
        self.kernel.grad = None
        self.padding = kernel_size // 2
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def __call__(self, img_output, img_target, device):
        if self.kernel.device != img_output.device:
            self.kernel = self.kernel.to(device)
        return self.GetSSIM(img_output, img_target)
        
    def GetSSIM(self, s, t):
        with torch.no_grad():
            # Compute means
            ux = conv2d(s, self.kernel, padding=self.padding, groups=3)
            uy = conv2d(t, self.kernel, padding=self.padding, groups=3)
            # Compute variances
            uxx = conv2d(s * s, self.kernel, padding=self.padding, groups=3)
            uyy = conv2d(t * t, self.kernel, padding=self.padding, groups=3)
            uxy = conv2d(s * t, self.kernel, padding=self.padding, groups=3)

            # print(ux.shape, uy.shape, uxx.shape, uyy.shape, uxy.shape)
            return ((2 * ux * uy + self.c1) * (2 * (uxy - ux * uy) + self.c2) 
                    / (ux ** 2 + uy ** 2 + self.c1) / ((uxx - ux * ux) + (uyy - uy * uy) + self.c2))
        

    def GetKernel(self, kernel_size: int, sigma: float) -> Tensor:
        start = (1 - kernel_size) / 2
        end = (1 + kernel_size) / 2
        kernel_1d = arange(start, end, step=1, dtype=float32)
        kernel_1d = exp(-pow(kernel_1d / sigma, 2) / 2)
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

        kernel_2d = matmul(kernel_1d.t(), kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
        return kernel_2d
    
    def __repr__(self):
        return "SSIM"
    
if __name__ == "__main__":
    ssim = SSIM()
    source = "/home/matsukawa_3/datasets/Bokeh_Simulation_Dataset/train/bokeh/0.jpg"
    target = "/home/matsukawa_3/datasets/Bokeh_Simulation_Dataset/train/original/0.jpg"
