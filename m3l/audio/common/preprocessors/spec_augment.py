# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch

import torchaudio


class SpecAugment(torchaudio.transforms.SpecAugment):
    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        if self.training:
            return super().forward(logx), y
        else:
            return logx, y
