# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class AttentionHead(nn.Module):
    """Attention-based pooling head for weak predictions in SED.

    Args:
        n_channels (int): Number of input feature channels.
        n_class (int): Number of output classes.

    Returns:
        torch.Tensor: Weak (clip-level) predictions of shape ``[B, n_class]``.
    """

    def __init__(self, n_channels: int, n_class: int):
        super().__init__()
        self.attention_head = nn.Sequential(nn.Conv1d(n_channels, n_class, 1), nn.Softmax(-2))

    def forward(self, h: torch.Tensor, y_strong: torch.Tensor):
        att = self.attention_head(h)
        return (att * y_strong).mean(-1) / att.mean(-1).clip(1e-6)
