# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class Normalize(nn.Module):
    """
    Normalize the log-mel spectrogram using running mean and power.
    Args:
        num_channels (int): Number of channels in the log-mel spectrogram.
        momentum (float): Momentum for updating running statistics. Default is 0.99.
    """

    def __init__(self, num_channels, momentum=0.99):
        super().__init__()

        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros([1, num_channels, 1]))
        self.register_buffer("running_power", torch.ones([1, num_channels, 1]))

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        if self.training:
            mean_ = logx.mean(axis=(0, 2), keepdims=True).detach()
            power_ = logx.square().mean(axis=(0, 2), keepdims=True).detach()

            torch.distributed.all_reduce(mean_, torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(power_, torch.distributed.ReduceOp.AVG)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_
            self.running_power = self.momentum * self.running_power + (1 - self.momentum) * power_

        scale = (self.running_power - self.running_mean**2).clip(1e-6).sqrt()

        return (logx - self.running_mean) / scale, y
