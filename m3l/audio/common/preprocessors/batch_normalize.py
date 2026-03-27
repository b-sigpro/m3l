# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class BatchNormalize(nn.Module):
    """Apply 2D batch normalization to the input tensor while keeping the label unchanged.

    Args:
        num_channels (int): Number of feature channels in the input tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            - Normalized tensor with the same shape as the input.
            - Unchanged label tensor (if provided).
    """

    def __init__(self, num_channels):
        super().__init__()

        self.bn = nn.BatchNorm2d(num_channels)

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        return self.bn(logx), y
