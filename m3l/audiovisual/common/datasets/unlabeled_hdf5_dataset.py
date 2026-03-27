# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
from tempfile import NamedTemporaryFile

# import time
import torch
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms as transforms

from aiaccel.torch.datasets import CachedDataset, RawHDF5Dataset


class AVDataset(Dataset):
    """Audio-Visual dataset loader with caching and temporary video decoding.

    Args:
        dataset_path (Path): Path to the HDF5 dataset file.
        sr (int): Audio sampling rate.
        fps (int): Frames per second for video sampling.
        train (bool, optional): Whether the dataset is for training. Defaults to False.
        ignore_error (bool, optional): Whether to ignore corrupted samples. Defaults to False.

    Methods:
        __len__: Returns the total number of samples in the dataset.
        __getitem__: Retrieves a single sample consisting of audio waveform and video frame.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - wav (torch.Tensor): Audio waveform tensor in ``bfloat16``.
            - frames (torch.Tensor): Video frame tensor normalized to ``[0, 1]`` in ``bfloat16``.
    """

    def __init__(
        self,
        dataset_path: Path,
        sr: int,
        fps: int,
        train: bool = False,
        ignore_error: bool = False,
    ):
        self.cached_dataset = CachedDataset(RawHDF5Dataset(dataset_path))

        self.sr = sr
        self.fps = fps

        self.ignore_error = ignore_error
        self.train = train

    def __len__(self):
        return len(self.cached_dataset)

    def __getitem__(self, index: int):
        sample = self.cached_dataset[index]

        wav = sample["audio"]
        wav = torch.as_tensor(wav, dtype=torch.bfloat16)

        # load video data (a temp file on tmpfs)
        with NamedTemporaryFile(dir="/dev/shm", suffix=".mp4") as f:
            f.write(sample["video"])
            frames, *_ = torchvision.io.read_video(
                f.name,
                start_pts=1 / 2,
                end_pts=1 / 2 + 0.2,
                pts_unit="sec",
                output_format="TCHW",
            )
        frames = frames[0].to(torch.bfloat16) / 255

        return wav, frames
