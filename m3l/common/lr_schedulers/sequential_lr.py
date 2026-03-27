# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import torch


class SequentialLR(torch.optim.lr_scheduler.SequentialLR):
    """Sequential learning rate scheduler with functional constructor.

    This class wraps :class:`torch.optim.lr_scheduler.SequentialLR`, but allows
    passing a list of functions instead of instantiated schedulers. Each
    function should take an optimizer and return a scheduler instance.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        schedulers_fn (list[Callable[[torch.optim.Optimizer], LRScheduler]]):
            List of functions that create schedulers.
        milestones (list[int]): List of epoch indices at which to switch to the next scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers_fn: list[Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]],
        milestones: list[int],
    ):
        super().__init__(optimizer, [fn(optimizer) for fn in schedulers_fn], milestones)
