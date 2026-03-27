# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Mapping

import torch


class LabelsGetter:
    """Utility class to extract label tensors from dataset samples.

    This class collects all tensor values from a sample (``Mapping``) except for
    specified excluded keys. By default, it excludes the key ``"boxes"``.

    Args:
        exclude_keys (list[str] | None, optional): Keys to exclude from label extraction.
            Defaults to ``["boxes"]``.

    Returns:
        tuple[torch.Tensor] | None: A tuple of label tensors extracted from the sample.
        Returns ``None`` if no tensors are found.
    """

    def __init__(self, exclude_keys: list[str] | None = None):
        if exclude_keys is None:
            self.exclude_keys = ["boxes"]
        else:
            self.exclude_keys = exclude_keys

    def __call__(self, sample):
        if isinstance(sample, (tuple, list)):
            sample = sample[1]

        assert isinstance(sample, Mapping)
        tensors = tuple(v for k, v in sample.items() if k.lower() not in self.exclude_keys and torch.is_tensor(v))
        if not tensors:
            return None
        return tensors
