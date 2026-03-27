# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import numpy as np

import torch
from torch import nn


class TimeRoll(nn.Module):
    """
    Randomly roll the input tensor along the time dimension during training.

    Args:
        None
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor | None):
        if self.training:
            *_, T = x.shape

            tau_unit = np.random.uniform(0, 1)

            tau = int(tau_unit * T)
            x = torch.roll(x, tau, dims=-1)

            if y is not None and y.dim() == 3:
                tau_y = int(tau_unit * y.shape[-1])
                y = torch.roll(y, tau_y, dims=-1)

        return x, y
