from random import Random
# from torchvision.transforms import functional as tf
# from torchvision.transforms.v2 import functional as tf

class BaseTransform(object):
    def __init__(self, seed=0):
        self.random = Random(seed)

    def __call__(self):
        self.UpdateParam()
    
    def UpdateParam(self, *args):
        # print("BaseTransform function")
        return