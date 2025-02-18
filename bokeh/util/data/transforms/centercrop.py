from util.data.transforms.basetransform import BaseTransform
from torchvision.transforms.v2.functional import center_crop

class CenterCrop(BaseTransform):
    def __init__(self, h=1024, w=1200, seed=0):
        super().__init__(seed)
        self.output_size = [h, w]

    def __call__(self, img_input, img_target):
        super().__call__()

        img_input = center_crop(img_input, self.output_size)
        img_target = center_crop(img_target, self.output_size)

        return img_input, img_target
    
    def UpdateParam(self):
        # print("centercrop function")
        return