# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import numpy as np

import torch

import lightning as lt


class ExponentialWarmupAnnealerCallback(lt.Callback):
    """Callback to exponentially anneal a parameter value during training.

    Args:
        name (str): Attribute name in the LightningModule to update.
        max_value (float): Maximum value for the parameter.
        duration (int): Number of steps over which to apply the annealing.
        exponent (float, optional): Exponent factor controlling the annealing curve. Defaults to -5.0.

    Methods:
        on_train_batch_start: Updates the parameter in the LightningModule at the beginning of each training batch.
    """

    def __init__(self, name: str, max_value: float, duration: int, exponent: float = -5.0):
        self.name = name
        self.max_value = max_value

        self.duration = duration
        self.exponent = exponent

    def on_train_batch_start(self, trainer: lt.Trainer, pl_module: Any, batch: torch.Tensor, batch_idx: torch.Tensor):
        x = np.clip(1 - trainer.global_step / self.duration, 0, 1)
        value = self.max_value * np.exp(self.exponent * x**2)

        setattr(pl_module, self.name, value)
