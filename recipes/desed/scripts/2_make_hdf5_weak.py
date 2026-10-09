#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aiaccel.torch.h5py.hdf5_writer import HDF5Writer

import librosa


class SEDHDF5Writer(HDF5Writer[Path, tuple[pd.DataFrame, list[str]]]):
    def __init__(self, split: str, dataset_path: Path, sample_rate: int):
        super().__init__()

        self.split = split

        self.dataset_path = dataset_path

        self.sample_rate = sample_rate

    def prepare_globals(self) -> tuple[list[Path], tuple[pd.DataFrame, list[str]]]:
        with open(self.dataset_path / "metadata" / "label_names.json") as f:
            label_list = json.load(f)

        match self.split:
            case "train":
                wav_filename_list = list((self.dataset_path / "audio" / "train" / "weak").glob("*.wav"))

                annotations = pd.read_table(self.dataset_path / "annotation" / "weak.tsv", index_col="filename")
            case _:
                raise ValueError(f'self.split must be "train", but {self.split} is given.')

        return wav_filename_list, (annotations, label_list)

    def prepare_group(
        self,
        item: Path,
        context: tuple[pd.DataFrame, list[str]],
    ) -> dict[str, dict[str, np.ndarray]]:
        wav_filename, (annotations, label_list) = item, context

        name = wav_filename.name

        # load wav
        wav, sr = librosa.load(wav_filename, sr=self.sample_rate, mono=True, dtype=np.float32)

        wav = np.r_[wav, np.zeros(max(0, self.sample_rate * 10 - len(wav)), dtype=np.float32)]
        wav = wav[: 10 * self.sample_rate]

        label_vec = np.zeros([10], dtype=np.int8)
        for label in annotations.loc[name]["event_labels"].split(","):
            lidx = label_list.index(label)

            label_vec[lidx] = 1

        return {name: {"wav": wav, "label": label_vec}}


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
    hdf_filename = dataset_path / "hdf5" / f"weak.{args_str}.hdf5"
    assert not hdf_filename.exists()

    writer = SEDHDF5Writer(args.split, dataset_path, args.sample_rate)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
