# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch
from torch import nn


class StrongWeakSEDModel(nn.Module):
    def __init__(self, encoder: nn.Module, strong_head: nn.Module, weak_head: nn.Module):
        super().__init__()

        self.encoder = encoder
        self.strong_head = strong_head
        self.weak_head = weak_head

    def forward(
        self, feats: dict[str, torch.Tensor], output_weak: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        h = self.encoder(feats)
        y_pred = self.strong_head(h)

        if output_weak:
            return y_pred, self.weak_head(h, y_pred)
        else:
            return y_pred
