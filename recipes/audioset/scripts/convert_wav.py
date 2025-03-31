#! /usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path

from rich.progress import track

import librosa as lr
import soundfile as sf


def main(args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("split", type=str, default="train")
    args = parser.parse_args()

    match args.split:
        case "train":
            src_fname_list = (
                list((dataset_path / "audios" / "unbalanced_train_segments").glob("*.wav")) + 
                list((dataset_path / "audios" / "balanced_train_segments").glob("*.wav"))
            )
        case "eval":
            src_fname_list = list((dataset_path / "audios" / "eval_segments").glob("*.wav"))
    src_fname_list.sort()

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        src_fname_list = src_fname_list[start:end]

    dst_path = dataset_path / args.split / "audios"
    dst_path.mkdir(parents=True, exist_ok=True)

    for src_fname in track(src_fname_list):
        try:
            wav, sr = lr.load(src_fname, sr=16000, mono=False, res_type="kaiser_best")
        except ValueError as e:
            print(f"Error: {e}")
        else:
            if len(wav.shape) == 2:
                wav = wav.T

            sf.write(f"{dst_path / src_fname.name}", wav, sr)


if __name__ == "__main__":
    main()
