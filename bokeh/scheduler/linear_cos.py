from torch.optim.lr_scheduler import _LRScheduler

class LinearCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, 
                 warm_epoch=1, max_epoch=50, last_epoch=0,
                 start=1.0e-5, goal=1.0e-5):
        """linear warmup -> cosine annealingを行うスケジューラ
        """
        super().__init__(optimizer, last_epoch)
        self.warm_epoch = warm_epoch
        self.max_epoch = max_epoch
        self.last_epoch = last_epoch
        self.start = start
        self.goal = goal

    def get_lr(self):
        if self.last_epoch <= self.warm_epoch:
            return [group["lr"] + (self.base_lrs[0] - self.start) / self.warm_epoch 
                    for group in self.optimizer.param_groups]
        