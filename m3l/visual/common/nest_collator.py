# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch.utils.data._utils.collate import default_collate


class NestedTensorCollator:
    """Collate function that converts selected keys into nested tensors.

    This collator is useful when batching variable-length tensors, such as
    sequences or images, while keeping other keys collated with the default
    PyTorch behavior.

    Args:
        nested_tensor_keys (Iterable[str]): Keys that should be collated into
            ``torch.nested.NestedTensor``.

    Methods:
        __call__(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            Collates a batch of mappings into a dictionary where selected keys
            are represented as ``NestedTensor``.

    Returns:
        dict[str, Any]: A dictionary containing collated batch data.
            - Keys in ``nested_tensor_keys`` -> ``torch.nested.NestedTensor``
            - Other keys -> Default PyTorch-collated values
    """

    def __init__(
        self,
        nested_tensor_keys: Iterable[str],
    ):
        self.nested_tensor_keys = set(nested_tensor_keys)

    def __call__(self, batch: Sequence[Mapping[str, Any]]):
        if not batch:
            raise ValueError("Empty batch")

        if not all(isinstance(sample, Mapping) for sample in batch):
            raise TypeError("All samples in the batch should be Mapping objects.")

        if any(key not in batch[0] for key in self.nested_tensor_keys):
            raise KeyError(f"Missing keys in the (1st) batch sample: {self.nested_tensor_keys - set(batch[0])}")

        collated: dict[str, Any] = {}
        for key in batch[0]:
            values = [sample[key] for sample in batch]
            if key in self.nested_tensor_keys:
                collated[key] = torch.nested.as_nested_tensor(values, layout=torch.strided)
            else:
                collated[key] = default_collate(values)

        return collated
