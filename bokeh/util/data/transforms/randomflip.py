from util.data.transforms.basetransform import BaseTransform
from torchvision.transforms.v2.functional import hflip, vflip

class RandomFlip(BaseTransform):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.flip_h = True
        self.flip_v = False

    def __call__(self, img_input, img_target):
        super().__call__()

        if self.flip_h:
            img_input = hflip(img_input)
            img_target = hflip(img_target)
        if self.flip_v:
            img_input = vflip(img_input)
            img_target = vflip(img_target)

        return img_input, img_target

    def UpdateParam(self):
        # print(f"randomflip function")
        self.flip_h = (0.5 <= self.random.random())
