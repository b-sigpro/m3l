# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import torch
from torch import nn

from m3l.common.nn.multiscale_deformable_attention import MultiScaleDeformableAttention
from m3l.common.nn.multiscale_deformable_transformer import (
    MultiScaleDeformableTransformerDecoderLayer,
    MultiScaleDeformableTransformerEncoderLayer,
)

from m3l.common.nn.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from m3l.common.nn.transformer import TransformerDecoderLayer, TransformerEncoderLayer


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


class LinearAct(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, activation: type[nn.Module] = nn.ReLU):
        super().__init__(nn.Linear(in_features, out_features), activation())


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, in_features: int, hidden_dim: int, n_layers: int, out_dim: int):
        super().__init__(
            LinearAct(in_features, hidden_dim),
            *[LinearAct(hidden_dim, hidden_dim) for _ in range(n_layers - 2)],
            nn.Linear(hidden_dim, out_dim),
        )


__all__ = [
    "MultiScaleDeformableAttention",
    "MultiScaleDeformableTransformerDecoderLayer",
    "MultiScaleDeformableTransformerEncoderLayer",
    "SinusoidalPositionalEmbedding",
    "TransformerDecoderLayer",
    "TransformerEncoderLayer",
]
