from numpy import mean, pow, log10, ndarray

class PSNR(object):
    def __init__(self):
        pass
        
    def __call__(self, o:ndarray, t:ndarray):
        mse = mean(pow((o - t), 2)) + 1.0e-6

        return 10 * log10(255.0 ** 2 / mse)
    
    def __repr__(self):
        return "PSNR"