# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from typing import Literal

import numpy as np
from scipy.stats import truncnorm

import torch
from torch import nn


class Scale(nn.Module):
    def __init__(self, normalize: Literal["std", "max", "none"] | bool = "std"):
        super().__init__()

        self.normalize = normalize

    def forward(self, wav: torch.Tensor):
        if self.normalize:
            if self.normalize == "std":
                wav = wav / wav.square().mean(dim=1, keepdim=True).sqrt().clip(1e-8)
            elif self.normalize == "max":
                wav = wav / wav.abs().amax(axis=1, keepdim=True).clip(1e-8)
            elif self.normalize == "none" or self.normalize is False:
                pass
            else:
                raise NotImplementedError()

        if self.training:
            B, _ = wav.shape

            scale = truncnorm(-np.log(3), np.log(3), 0, 1).rvs((B, 1))
            wav = wav * torch.tensor(scale, dtype=torch.float32, device=wav.device).exp()

        return wav
