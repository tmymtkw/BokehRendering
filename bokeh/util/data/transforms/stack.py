from random import Random
from util.data.transforms.basetransform import BaseTransform

class Stack(object):
    def __init__(self, seed=0):
        self.transforms = []
        # 乱数の生成器
        # 各transformに入力される共通の乱数を生成する
        self.random = Random(seed)

    def __call__(self, img_input, img_target):
        for transform in self.transforms:
            img_input, img_target = transform(img_input, img_target)

        return img_input, img_target
    
    def Push(self, transform):
        assert issubclass(type(transform), BaseTransform), \
            f"\n[ERROR] incorrect transform class: {type(transform)}"
        
        self.transforms.append(transform)