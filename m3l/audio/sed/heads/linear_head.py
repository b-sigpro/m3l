# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from torch import nn


class LinearHead(nn.Sequential):
    """Linear classification head for SED.

    Args:
        n_channels (int): Number of input feature channels.
        n_class (int): Number of output classes.
    """

    def __init__(self, n_channels: int, n_class: int):
        super().__init__(nn.Conv1d(n_channels, n_class, 1), nn.Sigmoid())
