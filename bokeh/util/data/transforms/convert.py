from util.data.transforms.basetransform import BaseTransform
from torch import float16, float32

class Convert(BaseTransform):
    def __init__(self, convert_type="float32"):
        if "16" in convert_type:
            self.type = float16
        else:
            self.type = float32

    def __call__(self, img_input, img_target):
        img_input = img_input.to(self.type)
        img_target = img_target.to(self.type)

        return img_input, img_target