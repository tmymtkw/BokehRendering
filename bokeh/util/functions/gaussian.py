from torch import Tensor, arange, exp, pow, matmul, float32

def getKernel(kernel_size: int, sigma: float) -> Tensor:
    start = (1 - kernel_size) / 2
    end = (1 + kernel_size) / 2
    kernel_1d = arange(start, end, step=1, dtype=float32)
    kernel_1d = exp(-pow(kernel_1d / sigma, 2) / 2)
    kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(dim=0)

    kernel_2d = matmul(kernel_1d.t(), kernel_1d)
    kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size).contiguous()
    return kernel_2d
