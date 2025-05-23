from typing import Any

import numpy as np

import torch

import lightning as lt


class ExponentialWarmupAnnealerCallback(lt.Callback):
    def __init__(self, name: str, max_value: float, duration: int, exponent: float = -5.0):
        self.name = name
        self.max_value = max_value

        self.duration = duration
        self.exponent = exponent

    def on_train_batch_start(self, trainer: lt.Trainer, pl_module: Any, batch: torch.Tensor, batch_idx: torch.Tensor):
        x = np.clip(1 - trainer.global_step / self.duration, 0, 1)
        value = self.max_value * np.exp(self.exponent * x**2)

        setattr(pl_module, self.name, value)
