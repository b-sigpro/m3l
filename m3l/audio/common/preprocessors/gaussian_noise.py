# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class GaussianNoise(nn.Module):
    """
    Add Gaussian noise to the log-mel spectrogram during training.

    Args:
        max_scale (float): Maximum scale of the Gaussian noise to be added. Default is 0.2.
    """

    def __init__(self, max_scale: float = 0.2):
        super().__init__()

        self.max_scale = max_scale

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        if self.training:
            B, *shape_ = logx.shape

            scale = self.max_scale * torch.rand([B] + [1] * len(shape_), device=logx.device)
            logx = logx + scale * torch.randn(logx.shape, device=logx.device)

        return logx, y
