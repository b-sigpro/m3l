from torch import nn
from torch.nn.common_types import _size_1_t, _size_2_t

from einops.layers.torch import Rearrange


class ConvBlock2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        pool_size: _size_2_t,
        dropout: float,
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding="same", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(pool_size),
            nn.Dropout(dropout),
        )


class AvgMaxPool1d(nn.Module):
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t,
        padding: _size_1_t = 0,
    ):
        super().__init__()

        self.max_pool = nn.MaxPool1d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, padding)

    def forward(self, x):
        return self.max_pool(x) + self.avg_pool(x)


class CNN14(nn.Sequential):
    def __init__(
        self,
        dim_latent: int = 512,
        dropout: float = 0.2,
        dropout_last: float = 0.5,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            Rearrange("b f t -> b 1 f t"),
            #
            ConvBlock2d(1, ch := 64, 3, 2, dropout),
            ConvBlock2d(ch, ch := 128, 3, 2, dropout),
            ConvBlock2d(ch, ch := 256, 3, 2, dropout),
            ConvBlock2d(ch, ch := 512, 3, 2, dropout),
            ConvBlock2d(ch, ch := 1024, 3, 2, dropout),
            ConvBlock2d(ch, ch := 2048, 3, 1, dropout),
            #
            nn.AdaptiveAvgPool2d((1, None)),
            Rearrange("b c 1 t -> b c t"),
            AvgMaxPool1d(3, 1, 1),
            #
            nn.Dropout(dropout_last),
            nn.Conv1d(2048, dim_latent, 1),
        )
