import numpy as np

import torch
from torch import nn


class Mixup(nn.Module):
    def __init__(self, beta: float = 0.2, hard: bool = False):
        super().__init__()

        self.beta = beta
        self.hard = hard

    def forward(self, x: torch.Tensor, y: torch.Tensor | None):
        if self.training and np.random.random() > 0.5:
            B, *_ = x.shape
            assert B >= 2 and B % 2 == 0

            perm = torch.arange(B).flip(-1)
            lm = torch.from_numpy(np.random.beta(self.beta, self.beta, size=[B])).to(x)
            x = torch.einsum("b,b...->b...", lm, x) + torch.einsum("b,b...->b...", 1 - lm, x[perm])

            if y is not None:
                if self.hard:
                    y = torch.maximum(y, y[perm])
                else:
                    y = torch.einsum("b,b...->b...", lm, y) + torch.einsum("b,b...->b...", 1 - lm, y[perm])

        return x, y
