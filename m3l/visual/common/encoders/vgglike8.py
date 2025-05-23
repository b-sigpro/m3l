from torch import nn
from torch.nn.common_types import _size_2_t


class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: _size_2_t = 3,
        padding: _size_2_t = 1,
    ):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ksize, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(out_ch, out_ch, ksize, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ConvPoolBlock(ConvBlock):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: _size_2_t = 3,
        padding: _size_2_t = 1,
        pool_size: _size_2_t = 2,
    ):
        super().__init__(in_ch, out_ch, ksize, padding)
        self.pool = nn.MaxPool2d(pool_size)


class VGGLike8(nn.Sequential):
    def __init__(self, dim_latent: int):
        super().__init__(
            #
            ConvPoolBlock(3, 64),
            ConvPoolBlock(64, 128),
            ConvPoolBlock(128, 256),
            #
            ConvBlock(256, dim_latent),
        )
