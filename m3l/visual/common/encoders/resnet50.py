# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable

from torch import nn

from torchvision.models import ResNet50_Weights, resnet50


class TorchvisionResNet50(nn.Sequential):
    """
    A ResNet-50 model from torchvision, modified to output a latent vector of a specified
    dimension. The model is initialized with pretrained weights, and certain layers can be frozen
    to prevent them from being updated during training.

    Args:
        dim_latent (int): The dimension of the output latent vector.
        norm_layer (Callable[..., nn.Module] | None): Optional normalization layer to use in the model.
        pretrained (bool): If True, initializes the model with pretrained weights. Defaults to True.
        frozen_layer_names (list[str] | None): List of layer names to freeze. If None, defaults to freezing
            the first few layers of the ResNet-50 model. Defaults to ["conv1", "bn1", "layer1", "layer2",
            "layer3", "layer4"]
    """

    def __init__(
        self,
        d_model: int,
        weights: str = "DEFAULT",
        norm_layer: Callable[..., nn.Module] | None = None,
        frozen_layer_names: list[str] | None = None,
    ):
        super().__init__()

        for name, param in resnet50(weights=ResNet50_Weights[weights], norm_layer=norm_layer).named_children():
            if name in ["avgpool", "fc"]:
                continue

            setattr(self, name, param)

        self.head = nn.Conv2d(2048, d_model, 1)

        if frozen_layer_names is None:
            frozen_layer_names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]

        for layer_name in frozen_layer_names:
            for param in getattr(self, layer_name).parameters():
                param.requires_grad = False
