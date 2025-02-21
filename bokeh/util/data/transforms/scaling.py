from .basetransform import BaseTransform
from torchvision.transforms.v2.functional import resize

class Scaling(BaseTransform):
    def __init__(self, size):
        assert len(size) == 1 or len(size) == 2, f"[ERROR] uncorrect scaling size : {size}"
        self.size = size

    def __call__(self, img_input, img_target):
        img_input = resize(img_input, self.size, antialias=True)
        img_target = resize(img_input, self.size, antialias=True)
        return img_input, img_target