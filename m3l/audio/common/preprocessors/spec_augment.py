import torch

import torchaudio


class SpecAugment(torchaudio.transforms.SpecAugment):
    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        if self.training:
            return super().forward(logx), y
        else:
            return logx, y
