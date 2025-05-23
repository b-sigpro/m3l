from argparse import ArgumentParser
from itertools import chain
import json
from math import ceil, floor, isnan
from pathlib import Path

from omegaconf import OmegaConf as oc

import numpy as np
import pandas as pd

from aiaccel.torch.h5py.hdf5_writer import HDF5Writer
from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config

import librosa


class SEDHDF5Writer(HDF5Writer[Path, tuple[pd.DataFrame, list[str]]]):
    def __init__(self, split: str, desed_path: Path, sample_rate: int, label_resolution: int):
        super().__init__()

        self.split = split

        self.desed_path = desed_path

        self.sample_rate = sample_rate
        self.label_resolution = label_resolution

    def prepare_globals(self) -> tuple[list[Path], tuple[pd.DataFrame, list[str]]]:
        with open(Path.cwd() / "metadata" / "label_names.json") as f:
            label_list = json.load(f)

        match self.split:
            case "train":
                audio_path = self.desed_path / "audio" / "train"
                wav_filename_list = list(
                    chain(
                        (audio_path / "strong_label_real").glob("*.wav"),
                        (audio_path / "synthetic21_train" / "soundscapes").glob("*.wav"),
                    )
                )

                metadata_path = self.desed_path / "metadata" / "train"
                annotations = pd.concat(
                    [
                        pd.read_table(metadata_path / "audioset_strong.tsv"),
                        pd.read_table(metadata_path / "synthetic21_train/soundscapes.tsv"),
                    ]
                )
            case "validation":
                audio_path = self.desed_path / "audio" / "validation"
                wav_filename_list = list(
                    chain(
                        (audio_path / "validation").glob("*.wav"), (audio_path / "synthetic21_validation").glob("*.wav")
                    )
                )

                metadata_path = self.desed_path / "metadata" / "validation"
                annotations = pd.concat(
                    [
                        pd.read_table(metadata_path / "validation.tsv"),
                        pd.read_table(metadata_path / "synthetic21_validation" / "soundscapes.tsv"),
                    ]
                )
            case _:
                raise ValueError(f'self.split must be "train" or "validation", but {self.split} is given.')

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

        wav = wav[: 10 * self.sample_rate]
        wav = np.r_[wav, np.zeros(self.sample_rate * 10 - len(wav), dtype=np.float32)]

        label_mat = np.zeros([10, 10 * 1000 // self.label_resolution], dtype=np.int8)
        for _, (_, start, end, label) in annotations.loc[annotations["filename"] == name].iterrows():
            if isnan(start):
                continue

            lidx = label_list.index(label)

            start_tidx = floor(start * 1000 / self.label_resolution)
            end_tidx = ceil(end * 1000 / self.label_resolution)

            label_mat[lidx, start_tidx:end_tidx] = 1

        return {name: {"wav": wav, "label": label_mat}}


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
    hdf_filename = dataset_path / "hdf5" / f"strong.{args_str}.hdf5"
    hdf_filename.unlink(missing_ok=True)

    writer = SEDHDF5Writer(args.split, Path(config.path.desed), args.sample_rate, args.label_resolution)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
