# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from torch import nn

from einops.layers.torch import Rearrange

from kornia.filters import MedianBlur


class MedianFilter(nn.Sequential):
    """1D median filter for temporal signals.

    This module applies a median blur along the temporal axis of the input
    tensor. Internally, the input is reshaped to 2D to apply Kornia's
    ``MedianBlur`` operation and then reshaped back.

    Args:
        filter_size (int): Size of the median filter kernel applied along
            the temporal dimension.

    Returns:
        torch.Tensor: Filtered tensor of shape ``[B, C, T]``, where
        ``B`` is the batch size, ``C`` is the channel dimension, and
        ``T`` is the temporal length.
    """

    def __init__(self, filter_size: int):
        super().__init__(Rearrange("b c t -> b c t 1"), MedianBlur((filter_size, 1)), Rearrange("b c t 1 -> b c t"))
