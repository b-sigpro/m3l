# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import numpy as np

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class ExponentialWarmupScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, max_lr: float, duration: int, exponent: float = -5.0):
        self.optimizer = optimizer

        self.max_lr = max_lr

        self.duration = duration
        self.exponent = exponent

        self._set_lr(0)
        self.count = 0

    def step(self):
        x = np.clip(1 - self.count / self.duration, 0, 1)
        lr = self.max_lr * np.exp(self.exponent * x**2)

        self._set_lr(lr)

        self.count += 1

    def _set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_state_dict(self, state_dict):
        vars(self).update(state_dict)

    def state_dict(self):
        return {key: value for key, value in vars(self).items() if key != "optimizer"}
