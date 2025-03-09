from .ssim_loss import SSIMLoss
from util.functions.gaussian import getKernel
from torch.nn.functional import interpolate, conv2d
from torch import mean, pow

class BlurredLoss(SSIMLoss):
    def __init__(self, kernel_size=[5, 9, 11], sigma=[2, 2.5, 3], ratio=0.85,
                 alpha=1.0, beta=1.0):
        super().__init__()
        self.kernels = [getKernel(kernel_size[i], sigma[i]) 
                        for i in range(len(kernel_size))]
        self.paddings = [kernel // 2 for kernel in kernel_size]
        self.n = len(kernel_size)
        self.is_safe = False
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

    def forward(self, img_output, img_target):
        # resize target
        _, _, h_input, w_input = img_output.shape

        if not self.is_safe:
            _, _, h_target, w_target = img_target.shape
            assert h_input / h_target == w_input / w_target, \
                f"[ERROR] incorrect ratio : {h_target} / {h_input} != {w_target} / {w_input}"
            self.is_safe = True

        blur_target = interpolate(img_target, (h_input, w_input), mode="bilinear")
        for kernel, padding in zip(self.kernels, self.paddings):
            k = kernel.to(device=blur_target.device)
            blur_target =  blur_target * (1 - self.ratio) + conv2d(blur_target, k, padding=padding, groups=3) * self.ratio

        return (mean(pow(img_output - blur_target, 2)) * self.alpha 
                + super().forward(img_output, blur_target) * self.beta)