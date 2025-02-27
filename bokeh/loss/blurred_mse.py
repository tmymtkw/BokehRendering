from .mse import MSELoss
from util.functions.gaussian import getKernel
from torch.nn.functional import interpolate, conv2d
# from torchvision.utils import save_image

class BlurredMSELoss(MSELoss):
    def __init__(self, kernel_size=[5, 9, 11], sigma=[2, 2.5, 3], alpha=0.85):
        super().__init__()

        self.kernels = [getKernel(kernel_size[i], sigma[i]) 
                        for i in range(len(kernel_size))]
        self.paddings = [kernel // 2 for kernel in kernel_size]
        self.n = len(kernel_size)
        # self.bit_start = pow(2, self.n)
        self.is_safe = False
        self.alpha = alpha

    def forward(self, img_output, img_target):
        # resize target
        # check
        _, _, h_input, w_input = img_output.shape

        if not self.is_safe:
            _, _, h_target, w_target = img_target.shape
            assert h_input / h_target == w_input / w_target, \
                f"[ERROR] incorrect ratio : {h_target} / {h_input} != {w_target} / {w_input}"
            self.is_safe = True

        # bit = self.bit_start
        blur_target = interpolate(img_target, (h_input, w_input), mode="bilinear")
        # cnt = 1
        for kernel, padding in zip(self.kernels, self.paddings):
            k = kernel.to(device=blur_target.device)
            blur_target =  blur_target * (1 - self.alpha) + conv2d(blur_target, k, padding=padding, groups=3) * self.alpha
            # save_image(blur_target, f"bokeh/outputs/blur_{num}_{cnt}.png")
            # cnt += 1

        return super().forward(img_output, blur_target)
