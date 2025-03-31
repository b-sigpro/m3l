# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from torch import nn

from einops.layers.torch import Rearrange

from kornia.filters import MedianBlur


class MedianFilter(nn.Sequential):
    def __init__(self, filter_size: int):
        super().__init__(Rearrange("b c t -> b c t 1"), MedianBlur((filter_size, 1)), Rearrange("b c t 1 -> b c t"))
