# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class TaggingModel(nn.Module):
    """Generic audio tagging model with an encoder-head structure.

    This model first processes input features with the given encoder,
    then applies a classification head on the resulting representation.

    Args:
        encoder (nn.Module): Feature encoder module that takes
            ``feats`` as input.
        head (nn.Module): Classification head module applied to the
            encoder outputs.

    Forward:
        Args:
            feats (dict[str, torch.Tensor]): Dictionary of input features,
                typically containing at least a spectrogram tensor.

        Returns:
            torch.Tensor: Model predictions with shape ``(B, n_class)``.
    """

    def __init__(self, encoder, head):
        super().__init__()

        self.encoder = encoder
        self.head = head

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(feats)
        return self.head(h)
