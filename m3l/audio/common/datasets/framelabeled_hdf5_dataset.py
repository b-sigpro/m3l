# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

import torch

from aiaccel.torch.datasets import HDF5Dataset


class FrameLabeledHDF5Dataset(HDF5Dataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:  # type: ignore
        item = super().__getitem__(index)

        wav = item["wav"]
        label = item["label"].to(torch.float32)

        return wav, label, self.grp_list[index]
