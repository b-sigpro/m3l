# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class StrongSEDModel(nn.Module):
    """Sound Event Detection (SED) model with a single strong head.

    This model uses an encoder to extract features and a strong head
    to predict frame-level labels.

    Args:
        encoder (nn.Module): Feature encoder module.
        head (nn.Module): Head module for strong (frame-level) predictions.

    Returns:
        torch.Tensor: Strong (frame-level) predictions.
    """

    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(feats)
        return self.head(h)
