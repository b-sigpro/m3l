from pathlib import Path

import torch
from torch.utils.data import Dataset

from aiaccel.torch.datasets import CachedDataset, HDF5Dataset


class FrameLabeledHDF5Dataset(Dataset):
    def __init__(self, dataset_path: Path | str, grp_list: Path | str | list[str] | None = None) -> None:
        self._dataset = CachedDataset(HDF5Dataset(dataset_path, grp_list))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:  # type: ignore
        item = self._dataset[index]

        wav = item["wav"]
        label = item["label"].to(torch.float32)

        return wav, label, self._dataset.dataset.grp_list[index]
