from torch import nn


class LinearHead(nn.Sequential):
    def __init__(self, n_channels: int, n_class: int):
        super().__init__(nn.Conv1d(n_channels, n_class, 1), nn.Sigmoid())
