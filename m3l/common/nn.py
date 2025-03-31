# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Any

import torch
from torch import nn


class TupleSequential(nn.ModuleList):
    def __init__(self, *args: nn.Module):
        super().__init__(args)

    def forward(self, *args: Any):
        for module in self:
            args = module(*args)

        return args


class SingleOutputGRU(nn.GRU):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        output, _ = super().forward(input)
        return output
