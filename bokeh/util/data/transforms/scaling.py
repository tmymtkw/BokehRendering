from torch import float16, float32
# from torch import dtype

class Scaling(object):
    def __init__(self, val=255.0, convert_type="float32"):
        self.val = val

        if "16" in convert_type:
            self.type = float16
        else:
            self.type = float32

    def __call__(self, img_input, img_target):
        img_input = img_input.to(self.type)
        img_target = img_target.to(self.type)

        if self.val != 1.0:
            img_input /= self.val
            img_target /= self.val

        return img_input, img_target