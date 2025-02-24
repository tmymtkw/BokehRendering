# from torchvision.transforms.v2.functional import normalize
from .basetransform import BaseTransform

class Normalize(BaseTransform):
    def __init__(self):
        pass
        # assert (
        #     isinstance(mean, (list, tuple, float)),
        #     f"\n[ERROR] incorrect normalize format: {type(mean)}"
        # )
        # assert(
        #     isinstance(std, (list, tuple, float)),
        #     f"\n[ERROR] incorrect normalize format: {type(std)}"
        # )
        
        # if isinstance(mean, float):
        #     self.mean = [mean for _ in range(3)]
        # else:
        #     self.mean = mean

        # if isinstance(std, float):
        #     self.std = [std for _ in range(3)]
        # else:
        #     self.std = std

    def __call__(self, img_input, img_target):
        # img_input = normalize(tensor=img_input, mean=self.mean, std=self.std, inplace=True)
        # img_target = normalize(tensor=img_target, mean=self.mean, std=self.std, inplace=True)
        img_input /= 255.0
        img_target /= 255.0

        return img_input, img_target