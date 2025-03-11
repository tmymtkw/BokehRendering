from torch.optim.lr_scheduler import LRScheduler
from math import cos, pi

class LinearCosineScheduler(LRScheduler):
    def __init__(self, optimizer, 
                 warm_steps=100, flag_epoch=10, max_epochs=50,
                 start_lr=1.0e-5, goal_lr=1.0e-5):
        """linear warmup -> cosine annealingを行うスケジューラ
        """
        self.warm_steps = warm_steps
        self.warm_count = 0
        super().__init__(optimizer=optimizer)
        self.flag_epoch = flag_epoch
        self.max_epochs = max_epochs
        self.goal_lr = goal_lr

        self.warm_lr = (self.base_lrs[0] - start_lr) / float(warm_steps)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

    def step(self, epoch = None):
        if self.last_epoch != -1 and self.warm_count < self.warm_steps:
            self._step_count += 1
            
            if epoch is None:
                self.last_epoch += 1
            else:
                self.last_epoch = epoch
        else:
            super().step(epoch)

    def get_lr(self):
        if self.warm_count <= self.warm_steps:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self.last_epoch <= self.flag_epoch:
            return self.base_lrs
        else:
            return [self.goal_lr + (base_lr - self.goal_lr) 
                    * (1 + cos((self.last_epoch - self.flag_epoch) / (self.max_epochs - self.flag_epoch) * pi)) / 2.0
                    for base_lr in self.base_lrs]
        
    def warm(self):
        if (self.warm_steps < self.warm_count):
            return
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] += self.warm_lr
        
        self.warm_count += 1