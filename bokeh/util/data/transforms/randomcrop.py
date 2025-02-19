from util.data.transforms.basetransform import BaseTransform
from torchvision.transforms.v2.functional import crop

class RandomCrop(BaseTransform):
    def __init__(self, height=1024, width=1024, seed=0):
        super().__init__(seed=seed)
        self.top = 0
        self.left = 0
        self.height = height
        self.width = width
        self.h = height
        self.w = width

    def __call__(self, img_input, img_target):
        _, self.h, self.w = img_input.size()

        super().__call__()

        img_input = crop(img_input, self.top, self.left, self.height, self.width)
        img_target = crop(img_target, self.top, self.left, self.height, self.width)
        
        return img_input, img_target
    
    def UpdateParam(self):
        # print("RandomCrop function")

        max_h = self.h - self.height
        max_w = self.w - self.width

        if max_h == 0:
            self.top = 0
        else:
            self.top = self.random.randrange(0, max_h)

        if max_w == 0:
            self.left = 0
        else:
            self.left = self.random.randint(0, max_w)

