# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import torch
from torch import nn

from m3l.visual.common.encoders.resnet50 import TorchvisionResNet50


class TorchvisionResNet50MultiScale(nn.Module):
    """Multi-scale ResNet-50 backbone with projection heads.

    This module extends a Torchvision ResNet-50 encoder to provide
    multi-scale feature maps from different layers. Projection
    heads are attached to selected layers to produce fixed-size
    feature representations.

    Args:
        d_model (int): Output feature dimension for projection heads.
        weights (str, optional): Pretrained weights for ResNet-50. Defaults to ``"DEFAULT"``.
        norm_layer (Callable[..., nn.Module] | None, optional): Normalization layer factory. Defaults to ``None``.
        frozen_layer_names (list[str] | None, optional): List of layer names to freeze. Defaults to ``None``.
        n_levels (int, optional): Number of feature levels to extract from the backbone. Defaults to ``4``.
        add_last (bool, optional):
            Whether to add the final identity layer for the deepest features. Defaults to ``True``.

    Returns:
        list[torch.Tensor]: A list of feature maps at multiple scales.
    """

    def __init__(
        self,
        d_model: int,
        weights: str = "DEFAULT",
        norm_layer: Callable[..., nn.Module] | None = None,
        frozen_layer_names: list[str] | None = None,
        n_levels: int = 4,
        add_last: bool = True,
    ):
        super().__init__()

        layer_info = {
            "conv1": 64,
            "bn1": 64,
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
            "last": 2048,
        }

        resnet50 = TorchvisionResNet50(d_model, weights, norm_layer, frozen_layer_names)
        self.resnet50 = nn.ModuleDict({name: module for name, module in resnet50.named_children()})
        self.resnet50.pop("head")

        if add_last:
            self.resnet50["last"] = nn.Identity()

        self.heads = nn.ModuleDict()
        for name in list(self.resnet50.keys())[-n_levels:]:
            if name == "last":
                cnv_kwargs = {"kernel_size": 3, "stride": 2, "padding": 1}
            else:
                cnv_kwargs = {"kernel_size": 1, "stride": 1, "padding": 0}

            self.heads[name] = nn.Sequential(
                nn.Conv2d(layer_info[name], d_model, **cnv_kwargs),
                nn.GroupNorm(32, d_model),
            )

        self.reset_parameters()

    def reset_parameters(self):
        for head in self.heads.values():
            nn.init.xavier_uniform_(head[0].weight, gain=1)
            nn.init.constant_(head[0].bias, 0)

    def forward(self, image: torch.Tensor):
        x = image

        outputs = []
        for name, module in self.resnet50.items():
            x = module(x)

            if name in self.heads:
                outputs.append(self.heads[name](x))

        return outputs
