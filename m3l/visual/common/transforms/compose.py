# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Sequence

import torch.nn as nn

from torchvision.transforms import v2


class Compose(v2.Compose):
    """Composes several transforms together.
    Args:
        transforms (Sequence[Callable]): list of transforms to compose.
    """

    def __init__(self, transforms: Sequence[Callable]) -> None:
        super().__init__(transforms)
        self.transforms = nn.ModuleList(self.transforms)
