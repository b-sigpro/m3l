# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch
from torch.utils.data import Dataset

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class UnlabeledHDF5Dataset(Dataset):
    """
    A PyTorch Dataset wrapper for loading unlabeled audio clips from an HDF5 file.

    This dataset uses `HDF5Dataset` wrapped with `CachedDataset` to efficiently
    access waveform tensors (`wav`). Unlike the labeled version, this class
    does not return labels - only the waveform and the corresponding group name.

    Args:
        dataset_path (Path | str): Path to the HDF5 dataset file.
        grp_list (Path | str | list[str] | None, optional):
            A list of group names, a path to a file containing group names,
            or `None` to use all groups. Defaults to None.

    Returns:
        tuple[torch.Tensor, str]:
            - wav (torch.Tensor): The waveform tensor.
            - group_name (str): The group name corresponding to the sample.
    """

    def __init__(self, dataset_path: Path | str, grp_list: Path | str | list[str] | None = None) -> None:
        self._dataset = CachedDataset(HDF5Dataset(dataset_path, grp_list))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:  # type: ignore
        item = self._dataset[index]

        wav = item["wav"]

        return wav, self._dataset.dataset.grp_list[index]
