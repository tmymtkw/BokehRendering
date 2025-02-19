from util.data.transforms.basetransform import BaseTransform
# from torchvision.transforms.functional import crop
from torchvision.transforms.v2.functional import crop

class Crop(BaseTransform):
    def __init__(self, top=0, left=0, height=1024, width=1024, seed=0):
        super().__init__(seed)
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img_input, img_target):
        img_input = crop(img_input, self.top, self.left, self.height, self.width)
        img_target = crop(img_target, self.top, self.left, self.height, self.width)

        return img_input, img_target
    
    def UpdateParam(self):
        # print("crop function")
        return
        
