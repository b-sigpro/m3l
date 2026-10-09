#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from aiaccel.torch.h5py.hdf5_writer import HDF5Writer

import librosa


class SEDHDF5Writer(HDF5Writer[Path, None]):
    def __init__(self, split: str, dataset_path: Path, sample_rate: int):
        super().__init__()

        self.split = split

        self.dataset_path = dataset_path

        self.sample_rate = sample_rate

    def prepare_globals(self) -> tuple[list[Path], None]:
        match self.split:
            case "train":
                audio_path = self.dataset_path / "audio" / "train"
                wav_filename_list = list((audio_path / "unlabel_in_domain").glob("*.wav"))
            case _:
                raise ValueError(f'self.split must be "train", but {self.split} is given.')

        return wav_filename_list, None

    def prepare_group(
        self,
        item: Path,
        context: None,
    ) -> dict[str, dict[str, np.ndarray]]:
        wav_filename = item

        name = wav_filename.name

        # load wav
        wav, sr = librosa.load(wav_filename, sr=self.sample_rate, mono=True, dtype=np.float32)

        wav = np.r_[wav, np.zeros(max(0, self.sample_rate * 10 - len(wav)), dtype=np.float32)]
        wav = wav[: 10 * self.sample_rate]

        return {name: {"wav": wav}}


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--parallel", action="store_true")
    args, _ = parser.parse_known_args()

    args_str = f"{args.sample_rate}hz.{args.split}"

    # write HDF5 file
    hdf_filename = dataset_path / "hdf5" / f"unsupervised.{args_str}.hdf5"
    assert not hdf_filename.exists()

    writer = SEDHDF5Writer(args.split, dataset_path, args.sample_rate)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
