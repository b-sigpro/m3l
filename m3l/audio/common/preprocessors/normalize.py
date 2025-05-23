import torch
from torch import nn


class Normalize(nn.Module):
    def __init__(self, num_channels, momentum=0.99):
        super().__init__()

        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros([1, num_channels, 1]))
        self.register_buffer("running_power", torch.ones([1, num_channels, 1]))

    def forward(self, logx: torch.Tensor, y: torch.Tensor | None = None):
        if self.training:
            mean_ = logx.mean(axis=(0, 2), keepdims=True).detach()
            power_ = logx.square().mean(axis=(0, 2), keepdims=True).detach()

            torch.distributed.all_reduce(mean_)
            torch.distributed.all_reduce(power_)

            world_size = torch.distributed.get_world_size()
            mean_ /= world_size
            power_ /= world_size

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean_
            self.running_power = self.momentum * self.running_power + (1 - self.momentum) * power_

        scale = (self.running_power - self.running_mean**2).clip(1e-6).sqrt()

        return (logx - self.running_mean) / scale, y
