#!/usr/bin/env python3


from argparse import ArgumentParser
from collections import defaultdict
import csv
from itertools import islice
from math import ceil, floor
from pathlib import Path

from omegaconf import OmegaConf as oc

import numpy as np

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config
from aiaccel.torch.h5py.hdf5_writer import HDF5Writer

import librosa


class AudioHDF5Writer(HDF5Writer[tuple[Path, list[tuple[int, float, float]]], None]):
    def __init__(self, split: str, dataset_path: Path, sample_rate: int, label_resolution: int):
        super().__init__()

        self.dataset_path = dataset_path
        self.split = split

        self.sample_rate = sample_rate
        self.label_resolution = label_resolution

    def prepare_globals(self) -> tuple[list[tuple[Path, list[tuple[int, float, float]]]], None]:
        with open(self.dataset_path / "metadata" / "strong" / "mid_to_display_name.tsv") as f:
            reader = csv.reader(f, delimiter="\t")
            label_list = [label for label, _ in reader]

        wav_filename_dict = {}
        for split_name in (
            ["eval_segments"] if self.split == "eval" else ["balanced_train_segments", "unbalanced_train_segments"]
        ):
            with open(self.dataset_path / "metadata" / "weak" / f"{split_name}.csv") as f:
                reader = csv.reader(f, skipinitialspace=True)
                for yt_id, start, end, _ in islice(reader, 3, None):
                    name = f"Y{yt_id}_{float(start):.3f}_{float(end):.3f}.wav"
                    wav_filename_dict[yt_id] = self.dataset_path / "raw" / "audios" / split_name / name

        # load label info
        label_info_dict = defaultdict(list)
        with open(self.dataset_path / "metadata" / "strong" / f"audioset_{self.split}_strong.tsv") as f:
            reader = csv.reader(f, delimiter="\t")
            for segment_id, start_time, end_time, label in islice(reader, 1, None):
                filename = wav_filename_dict[segment_id[:11]]

                label_info_dict[filename].append((label_list.index(label), float(start_time), float(end_time)))

        label_info_list = [(filename, labels) for filename, labels in label_info_dict.items()]

        return label_info_list, None

    def prepare_group(
        self,
        item: tuple[Path, list[tuple[int, float, float]]],
        context: None,
    ) -> dict[str, dict[str, np.ndarray]]:
        wav_filename, labels = item

        # load wav
        if not wav_filename.exists():
            return {}

        wav, sr = librosa.load(wav_filename, sr=self.sample_rate, dtype="float32")

        if wav.shape[0] < 10 * sr:
            pad = np.zeros([10 * sr - wav.shape[0]], dtype=np.float32)
            wav = np.concatenate([wav, pad])

        wav = wav[: 10 * sr]

        label_mat = np.zeros([456, 10 * 1000 // self.label_resolution], dtype=np.int8)
        for lidx, start, end in labels:
            start_tidx = floor(start * 1000 / self.label_resolution)
            end_tidx = ceil(end * 1000 / self.label_resolution)

            label_mat[lidx, start_tidx:end_tidx] = 1

        return {wav_filename.name: {"wav": wav, "label": label_mat}}


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    # load config
    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--label_resolution", type=int, default=64)
    parser.add_argument("--parallel", action="store_true")
    args, unk_args = parser.parse_known_args()

    args_str = f"{args.sample_rate}hz_{args.label_resolution}ms.{args.split}"

    base_config = oc.merge(oc.from_cli(unk_args))
    config = load_config(dataset_path / "config.yaml", base_config)
    print_config(config)

    # write HDF5 file
    hdf_filename = dataset_path / "hdf5" / f"audio_strong.{args_str}.hdf5"
    hdf_filename.unlink(missing_ok=True)

    writer = AudioHDF5Writer(args.split, Path.cwd(), args.sample_rate, args.label_resolution)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
