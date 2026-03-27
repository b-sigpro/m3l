# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class StrongWeakSEDModel(nn.Module):
    """Sound Event Detection (SED) model with strong and weak heads.

    This model uses a shared encoder and two separate heads:
    one for strong labels (frame-level predictions) and another
    for weak labels (clip-level predictions). It supports flexible
    output depending on the task requirements.

    Args:
        encoder (nn.Module): Feature encoder module.
        strong_head (nn.Module): Head module for strong (frame-level) predictions.
        weak_head (nn.Module): Head module for weak (clip-level) predictions.

    Returns:
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            - If ``output_weak=False``: Strong (frame-level) predictions.
            - If ``output_weak=True``: A tuple containing
              (strong predictions, weak predictions).
    """

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
