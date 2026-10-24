# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as fn

from einops.layers.torch import Rearrange

from m3l.audio.common.encoders.cnn14 import ConvBlock2d


class CNNBEATsEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_emb: int = 768,
        dropout: float = 0.2,
        dropout_emb: float = 0.5,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            Rearrange("b f t -> b 1 f t"),
            #
            ConvBlock2d(1, ch := 64, 3, 2, dropout),
            ConvBlock2d(ch, ch := 128, 3, 2, dropout),
            ConvBlock2d(ch, ch := 256, 3, 2, dropout),
            ConvBlock2d(ch, ch := 512, 3, 1, dropout),
            ConvBlock2d(ch, ch := 1024, 3, 1, dropout),
            ConvBlock2d(ch, ch := 1024, 3, 1, dropout),
            #
            nn.AdaptiveAvgPool2d((1, None)),
            Rearrange("b c 1 t -> b c t"),
        )

        self.lin_emb = nn.Sequential(nn.Conv1d(1024 + dim_emb, d_model, 1), nn.Dropout(dropout_emb))

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.cnn(feats["logx"])

        h_emb = rearrange(feats["emb"], "b t c -> b c t")
        h_emb = fn.adaptive_avg_pool1d(h_emb, h.shape[-1])  # [B, C, T]

        return self.lin_emb(torch.concat((h, h_emb), dim=1))
