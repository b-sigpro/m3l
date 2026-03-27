# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as fn


class NormalizeBoundingBoxes(Transform):
    """Normalize bounding box coordinates to the range [0, 1].

    This transform divides the bounding box coordinates by the
    image width and height, effectively normalizing them with respect
    to the canvas size.

    Args:
        None

    Returns:
        tv_tensors.BoundingBoxes | Any: Normalized bounding boxes if the input
        is of type ``tv_tensors.BoundingBoxes``; otherwise, returns the input unchanged.
    """

    def make_params(self, flat_inputs):
        H, W = fn.get_size(flat_inputs[0])
        return {"H": H, "W": W}

    def transform(self, inpt, params):
        if not isinstance(inpt, tv_tensors.BoundingBoxes):
            return inpt

        boxes = inpt.to(dtype=torch.float32)
        boxes[..., 0::2] /= params["W"]
        boxes[..., 1::2] /= params["H"]
        boxes.canvas_size = (1, 1)

        return boxes
