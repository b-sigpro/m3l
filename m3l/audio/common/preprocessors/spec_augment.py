# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch

import torchaudio


class SpecAugment(torchaudio.transforms.SpecAugment):
    """
    SpecAugment for audio data.
    Args:
        time_mask_param (int): Maximum width of the time mask.
        freq_mask_param (int): Maximum width of the frequency mask.
        num_time_masks (int): Number of time masks to apply.
        num_freq_masks (int): Number of frequency masks to apply.
        p (float): Probability of applying SpecAugment. Default is 0.5.
    """

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        if self.training:
            return super().forward(logx), y
        else:
            return logx, y
