# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from collections.abc import Callable, Sequence

import torch.nn as nn

from torchvision.transforms import v2


class RandomChoice(v2.RandomChoice):
    """Randomly applies one of the given transforms.
    Args:
        transforms (Sequence[Callable[..., Any]]): List of transforms to choose from.
        p (list[float] | None, optional): Probabilities associated with each transform.
            If ``None``, each transform is chosen with equal probability.
            Defaults to ``None``.
    """

    def __init__(self, transforms: Sequence[Callable[..., Any]], p: list[float] | None = None) -> None:
        super().__init__(transforms, p)
        self.transforms = nn.ModuleList(self.transforms)
