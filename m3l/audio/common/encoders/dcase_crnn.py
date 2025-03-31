# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as fn

from einops.layers.torch import Rearrange

from m3l.common.nn import SingleOutputGRU


class GLU(nn.Conv2d):
    def __init__(self, io_channels: int):
        super().__init__(io_channels, io_channels, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return fn.sigmoid(input) * super().forward(input)


class Conv2dBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, pool_size: tuple[int, int], dropout: float):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm2d(out_channels),
            GLU(out_channels),
            nn.Dropout(dropout),
            nn.AvgPool2d(pool_size),
        )


class CRNN(nn.Module):
    def __init__(
        self,
        dropout: float = 0.5,
        dropout_rnn: float = 0.0,
        dim_emb: int | None = None,
        dropout_emb: float = 0.5,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            Rearrange("b (c f) t -> b c f t", c=1),
            Conv2dBlock(1, 16, (2, 2), dropout),  # F=64
            Conv2dBlock(16, 32, (2, 2), dropout),  # F=32
            Conv2dBlock(32, 64, (2, 1), dropout),  # F=16
            Conv2dBlock(64, 128, (2, 1), dropout),  # F=8
            Conv2dBlock(128, 128, (2, 1), dropout),  # F=4
            Conv2dBlock(128, 128, (2, 1), dropout),  # F=2
            Conv2dBlock(128, 128, (2, 1), dropout),  # F=1
            Rearrange("b c 1 t -> b c t"),
        )

        self.dim_emb = dim_emb
        if dim_emb is not None:
            self.lin_emb = nn.Sequential(nn.Conv1d(128 + dim_emb, 128, 1), nn.Dropout(dropout_emb))

        self.gru = nn.Sequential(
            Rearrange("b c t -> b t c"),
            SingleOutputGRU(128, 128, 2, batch_first=True, dropout=dropout_rnn, bidirectional=True),
            Rearrange("b t c -> b c t"),
            nn.Dropout(dropout),
        )

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.cnn(feats["logx"])

        if self.dim_emb is not None:
            h_emb = rearrange(feats["emb"], "b t c -> b c t")
            h_emb = fn.adaptive_avg_pool1d(h_emb, h.shape[-1])  # [B, C, T]

            h = self.lin_emb(torch.concat((h, h_emb), dim=1))

        h = self.gru(h)

        return h
