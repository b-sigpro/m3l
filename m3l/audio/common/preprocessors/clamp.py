# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch
from torch import nn


class Clamp(nn.Module):
    def __init__(self, value: float = 6):
        super().__init__()
        self.value = value

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        return logx.clip(-self.value, self.value), y
