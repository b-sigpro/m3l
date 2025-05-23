#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path

from omegaconf import DictConfig

import numpy as np

from aiaccel.torch.h5py.hdf5_writer import HDF5Writer
from aiaccel.utils import load_config, overwrite_omegaconf_dumper, print_config
import cv2

import soundfile as sf


class AudioVisualHDF5Writer(HDF5Writer[Path, None]):
    def __init__(self, split: str, dataset_path: Path, fps: int):
        super().__init__()

        self.audio_path = dataset_path / split / "audios"
        self.video_path = dataset_path / split / f"videos.{fps}fps"
        self.split = split
        self.fps = fps

    def prepare_globals(self) -> tuple[list[Path], None]:
        audio_filename_list = sorted(self.audio_path.glob("*.wav"))

        return audio_filename_list, None

    def prepare_group(
        self,
        item: Path,
        context: None,
    ) -> dict[str, dict[str, np.ndarray]]:
        wav_filename = item
        mp4_filename = self.video_path / f"{wav_filename.stem}.mp4"

        audio, sr = sf.read(wav_filename, dtype=np.float32, always_2d=True)
        audio = audio.mean(axis=-1)
        if audio.shape[0] < 9.95 * 16000:
            return {}
        audio = np.r_[audio, np.zeros(max(0, 16000 * 10 - len(audio)))]
        audio = audio[: 10 * 16000]

        if not os.path.exists(mp4_filename):
            return {}

        # check # of frames
        vc = cv2.VideoCapture(mp4_filename)
        n_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        vc.release()

        if n_frames < self.fps * 10:
            return {}

        # load as byte data
        with open(mp4_filename, "rb") as vf:
            video = vf.read()

        return {wav_filename.stem: {"audio": audio, "video": video}}


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    # load config
    overwrite_omegaconf_dumper()
    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    fps = config.convert.fps
    image_size = config.convert.image_size

    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("--parallel", action="store_true")
    args, unk_args = parser.parse_known_args()

    args_str = f"{args.split}.{image_size}image_size.{fps}fps"

    # write HDF5 file
    hdf_filename = dataset_path / "hdf5" / f"audiovisual_unsupervised.{args_str}.hdf5"
    hdf_filename.unlink(missing_ok=True)

    writer = AudioVisualHDF5Writer(args.split, dataset_path, fps)
    writer.write(hdf_filename, parallel=args.parallel)


if __name__ == "__main__":
    main()
