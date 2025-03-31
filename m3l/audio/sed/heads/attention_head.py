# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch
from torch import nn


class AttentionHead(nn.Module):
    def __init__(self, n_channels: int, n_class: int):
        super().__init__()

        self.attention_head = nn.Sequential(nn.Conv1d(n_channels, n_class, 1), nn.Softmax(-2))

    def forward(self, h: torch.Tensor, y_strong: torch.Tensor):
        att = self.attention_head(h)

        return (att * y_strong).mean(-1) / att.mean(-1).clip(1e-6)
