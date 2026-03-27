# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from torchvision import tv_tensors
from torchvision.transforms.v2 import ConvertBoundingBoxFormat as tvConvertBoundingBoxFormat
from torchvision.transforms.v2 import functional as fn


class ConvertBoundingBoxesFormat(tvConvertBoundingBoxFormat):
    """Bounding box format converter compatible with Torchvision v2.

    This class extends :class:`torchvision.transforms.v2.ConvertBoundingBoxFormat`
    to correctly handle ``tv_tensors.BoundingBoxes``.

    Args:
        format (str): Target bounding box format (e.g., ``"XYXY"``, ``"CXCYWH"``).

    Returns:
        tv_tensors.BoundingBoxes | Any: Converted bounding boxes if input is of type
        ``tv_tensors.BoundingBoxes``, otherwise returns the input unchanged.
    """

    def __init__(self, format):
        super().__init__(format)

    def transform(self, inpt, params):
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            return fn.convert_bounding_box_format(inpt, new_format=self.format)
        return inpt
