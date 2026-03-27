# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import random

from torchvision.transforms.v2 import RandomCrop
from torchvision.transforms.v2._utils import query_size


class RangeRandomCrop(RandomCrop):
    """Randomly crop an image within a specified range of crop sizes.

    The crop size is randomly chosen between ``min_crop`` and ``max_crop``
    for both height and width, subject to the original image dimensions.

    Args:
        min_crop (int): Minimum crop size.
        max_crop (int): Maximum crop size.

    Returns:
        Any: A randomly cropped image or tensor, depending on the input type.
    """

    def __init__(self, min_crop: int, max_crop: int):
        super().__init__(size=1)
        self.min_crop = min_crop
        self.max_crop = max_crop

    def make_params(self, flat_inputs):
        h, w = query_size(flat_inputs)
        out_h = random.randint(min(h, self.min_crop), min(h, self.max_crop))
        out_w = random.randint(min(w, self.min_crop), min(w, self.max_crop))

        return dict(
            needs_crop=True,
            top=random.randint(0, h - out_h),
            left=random.randint(0, w - out_w),
            height=out_h,
            width=out_w,
            needs_pad=False,
            padding=[0, 0, 0, 0],
        )
