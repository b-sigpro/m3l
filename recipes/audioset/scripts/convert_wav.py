#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
import os
from pathlib import Path

from rich.progress import track

import librosa as lr
import soundfile as sf


def convert_wav(src_filename: Path, dst_path: Path):
    try:
        wav, sr = lr.load(src_filename, sr=16000, mono=True, res_type="kaiser_best")
    except ValueError as e:
        print(f"ValueError: {e}")
    else:
        sf.write(f"{dst_path / src_filename.name}", wav, sr)


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("split", type=str, default="train")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    base_src_path = dataset_path / "raw" / "audios"
    match args.split:
        case "train":
            src_filename_list = list((base_src_path / "unbalanced_train_segments").glob("*.wav")) + list(
                (base_src_path / "balanced_train_segments").glob("*.wav")
            )
        case "eval":
            src_filename_list = list((base_src_path / "eval_segments").glob("*.wav"))
        case _:
            raise ValueError()

    src_filename_list.sort()

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        src_filename_list = src_filename_list[start:end]

    dst_path = dataset_path / args.split / "audios"
    dst_path.mkdir(parents=True, exist_ok=True)

    convert_wav_ = partial(convert_wav, dst_path=dst_path)
    with Pool(processes=args.processes) as p:
        for _ in track(p.imap_unordered(convert_wav_, src_filename_list)):
            pass


if __name__ == "__main__":
    main()
