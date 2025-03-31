# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import numpy as np

import torch
from torch import nn


class TimeRoll(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor | None):
        if self.training:
            *_, T = x.shape

            tau = np.random.randint(0, T)
            x = torch.roll(x, tau, dims=-1)

            if y is not None and y.dim() == 3:
                y = torch.roll(y, tau, dims=-1)

        return x, y
