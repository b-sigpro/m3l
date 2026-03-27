# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class Clamp(nn.Module):
    """
    Clamp the values of the log-mel spectrogram to a specified range.

    Args:
        value (float): The value to clamp the log-mel spectrogram values to. Default is 6.
    """

    def __init__(self, value: float = 6):
        super().__init__()
        self.value = value

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        return logx.clip(-self.value, self.value), y
