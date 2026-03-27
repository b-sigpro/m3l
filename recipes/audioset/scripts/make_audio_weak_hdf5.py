#!/usr/bin/env python3


# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import csv
from itertools import islice
from pathlib import Path

from omegaconf import OmegaConf as oc

import numpy as np

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config
from aiaccel.torch.h5py.hdf5_writer import HDF5Writer

import librosa


class AudioWeakHDF5Writer(HDF5Writer[tuple[Path, list[tuple[int, float, float]]], None]):
    def __init__(self, split: str, dataset_path: Path, sample_rate: int):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        self.sample_rate = sample_rate

    def prepare_globals(self) -> tuple[list[tuple[Path, list[int]]], None]:
        mid_idx_dict = {}
        with open(self.dataset_path / "metadata" / "weak" / "class_labels_indices.csv", newline="") as f:
            reader = csv.DictReader(f)
            mid_idx_dict = {row["mid"].strip(): int(row["index"].strip()) for row in reader}

        label_info_list = []
        for split_name in (
            ["eval_segments"] if self.split == "eval" else ["balanced_train_segments", "unbalanced_train_segments"]
        ):
            with open(self.dataset_path / "metadata" / "weak" / f"{split_name}.csv") as f:
                reader = csv.reader(f, skipinitialspace=True)
                for yt_id, start, end, labels in islice(reader, 3, None):
                    name = f"Y{yt_id}_{float(start):.3f}_{float(end):.3f}.wav"
                    filename = self.dataset_path / "raw" / "audios" / split_name / name

                    cls_ids = [mid_idx_dict[lbl] for lbl in labels.split(",")]
                    label_info_list.append((filename, cls_ids))

        return label_info_list, None

    def prepare_group(
        self,
        item: tuple[Path, list[int]],  # weak: (wav_path, [cls_id, ...])
        context: None,
    ) -> dict[str, dict[str, np.ndarray]]:
        wav_filename, labels = item

        # load wav
        if not wav_filename.exists():
            return {}

        wav, sr = librosa.load(wav_filename, sr=self.sample_rate, mono=True, dtype="float32")

        if wav.shape[0] < 10 * sr:
            pad = np.zeros([10 * sr - wav.shape[0]], dtype=np.float32)
            wav = np.concatenate([wav, pad])

        wav = wav[: 10 * sr]

        # for weak label
        label_mat = np.zeros([527], dtype=np.int8)
        for lidx in labels:
            label_mat[lidx] = 1

        return {wav_filename.name: {"wav": wav, "label": label_mat}}


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    # load config
    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--parallel", action="store_true")
    args, unk_args = parser.parse_known_args()

    args_str = f"{args.sample_rate}hz.{args.split}"

    base_config = oc.merge(oc.from_cli(unk_args))
    config = load_config(dataset_path / "config.yaml", base_config)
    print_config(config)

    # write HDF5 file
    hdf_filename = dataset_path / "hdf5" / f"audio_weak.{args_str}.hdf5"
    hdf_filename.unlink(missing_ok=True)

    writer = AudioWeakHDF5Writer(args.split, Path.cwd(), args.sample_rate)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
