from scripts.trainer import Trainer

class Validator(Trainer):
    def __init__(self):
        super().__init__()

    def Validate(self, epoch):
        super().Validate(epoch=epoch)