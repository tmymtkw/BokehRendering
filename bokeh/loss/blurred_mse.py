from .mse import MSELoss
from util.functions.gaussian import getKernel
from torch.nn.functional import interpolate, conv2d

class BlurredMSELoss(MSELoss):
    def __init__(self, kernel_size=[3, 5, 7], sigma=[0.3, 0.6, 1]):
        super().__init__()

        self.kernels = [getKernel(kernel_size[i], sigma[i]) 
                        for i in range(len(kernel_size))]
        self.paddings = [kernel // 2 for kernel in kernel_size]
        for kernel in self.kernels:
            kernel.grad = None
        self.n = len(kernel_size)
        self.bit_start = pow(2, self.n)
        self.is_safe = False

    def forward(self, img_input, img_target):
        # resize target
        # check
        _, _, h_input, w_input = img_input.shape

        if not self.is_safe:
            _, _, h_target, w_target = img_target.shape
            assert h_input / h_target == w_input / w_target, \
                f"[ERROR] incorrect ratio : {h_target} / {h_input} != {w_target} / {w_input}"
            self.is_safe = True

        bit = self.bit_start
        resized_target = interpolate(img_target, (h_input, w_input), mode="bilinear")
        blur_target = resized_target / bit

        for kernel, padding in zip(self.kernels, self.paddings):
            blur_target += conv2d(resized_target, kernel, padding=padding, groups=3) / bit
            bit *= 2

        return super().forward(img_input, blur_target)
