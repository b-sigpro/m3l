import torch
from torch import nn


class StrongSEDModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()

        self.encoder = encoder
        self.head = head

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.encoder(feats)
        return self.head(h)
