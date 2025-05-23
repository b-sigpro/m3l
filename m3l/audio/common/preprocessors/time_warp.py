import numpy as np

import torch
from torch import nn

from torchvision.transforms.functional import resize


class TimeWarp(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor | None):
        if self.training:
            _, F, T = x.shape

            t_src_unit = np.random.uniform(0.4, 0.6)
            t_dst_unit = t_src_unit + np.random.uniform(-0.1, +0.1)

            t_src, t_dst = int(t_src_unit * T), int(t_dst_unit * T)
            x = torch.cat([resize(x[..., :t_src], [F, t_dst]), resize(x[..., t_src:], [F, T - t_dst])], dim=-1)

            if y is not None and y.dim() == 3:
                _, L, Ty = y.shape

                t_src, t_dst = int(t_src_unit * Ty), int(t_dst_unit * Ty)
                y = torch.cat([resize(y[..., :t_src], [L, t_dst]), resize(y[..., t_src:], [L, T - t_dst])], dim=-1)

        return x, y
