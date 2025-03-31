# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import numpy as np

import torch
from torch import nn

from torchvision.transforms.functional import resize


class TimeWarp(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor | None):
        if self.training:
            _, F, T = x.shape

            t_src = np.random.randint(int(0.4 * T), int(0.6 * T))
            t_dst = t_src + np.random.randint(int(-0.1 * T), int(+0.1 * T))

            x = torch.cat([resize(x[..., :t_src], [F, t_dst]), resize(x[..., t_src:], [F, T - t_dst])], dim=-1)

            if y is not None and y.dim() == 3:
                _, L, _ = y.shape
                y = torch.cat([resize(y[..., :t_src], [L, t_dst]), resize(y[..., t_src:], [L, T - t_dst])], dim=-1)

        return x, y
