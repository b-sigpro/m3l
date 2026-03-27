# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from einops.layers.torch import Rearrange


class ConvBlock2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        pool_size: _size_2_t,
        dropout: float,
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(pool_size),
            nn.Dropout(dropout),
        )


class GlobalAvgMaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=-1) + x.amax(dim=-1)


class CNN14(nn.Sequential):
    """CNN14 architecture for audio classification.

    Based on PANNs (Pretrained Audio Neural Networks), this model applies
    convolutional blocks with pooling and dropout, followed by fully
    connected layers. The head outputs either sigmoid probabilities or
    raw logits.

    Args:
        n_class (int): Number of output classes.
        dropout (float, optional): Dropout rate for intermediate layers.
            Defaults to 0.2.
        dropout_last (float, optional): Dropout rate for final layers.
            Defaults to 0.5.
    """

    def __init__(
        self,
        n_class: int,
        dropout: float = 0.2,
        dropout_last: float = 0.5,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            Rearrange("b f t -> b 1 f t"),
            #
            ConvBlock2d(1, ch := 64, 3, 2, dropout),
            ConvBlock2d(ch, ch := 128, 3, 2, dropout),
            ConvBlock2d(ch, ch := 256, 3, 2, dropout),
            ConvBlock2d(ch, ch := 512, 3, 2, dropout),
            ConvBlock2d(ch, ch := 1024, 3, 2, dropout),
            ConvBlock2d(ch, ch := 2048, 3, 1, dropout),
            #
            nn.AdaptiveAvgPool2d((1, None)),
            Rearrange("b c 1 t -> b c t"),
            GlobalAvgMaxPool(),
            #
            nn.Dropout(dropout_last),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_last),
        )

        self.head = nn.Linear(2048, n_class)

    def forward(self, feat: dict[str, torch.Tensor], return_logits: bool = False) -> torch.Tensor:
        logits = super().forward(feat["logx"])

        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)
