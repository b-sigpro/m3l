#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
import subprocess

from aiaccel.config import prepare_config
from aiaccel.job.utils import split_tasks
from rich.progress import track


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("split", type=str, default="train")
    args = parser.parse_args()

    config = prepare_config(
        dataset_path / "config.yaml",
        print_config=True,
    )

    fps = config.convert.fps
    image_size = config.convert.image_size

    match args.split:
        case "train":
            src_filename_list = list(
                (dataset_path / "raw" / "videos" / "unbalanced_train_segments").glob("*.mp4")
            ) + list((dataset_path / "raw" / "videos" / "balanced_train_segments").glob("*.mp4"))
        case "eval":
            src_filename_list = list((dataset_path / "raw" / "videos" / "eval_segments").glob("*.mp4"))
        case _:
            raise ValueError(f"Unexpected split: {args.split}")
    src_filename_list.sort()

    src_filename_list = split_tasks(src_filename_list)

    dst_path = dataset_path / args.split / f"videos.{fps}fps"
    dst_path.mkdir(parents=True, exist_ok=True)

    # https://stackoverflow.com/questions/56628663/how-to-set-short-side-of-video-to-a-constant
    for src_fname in track(src_filename_list):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                src_fname,
                "-vf",
                (
                    rf"scale=w=iw*min({image_size}/iw\,{image_size}/ih):h=ih*min({image_size}/iw\,{image_size}/ih),"
                    rf"pad={image_size}:{image_size}:({image_size}-iw*min({image_size}/iw\,"
                    rf"{image_size}/ih))/2:({image_size}-ih*min({image_size}/iw\,{image_size}/ih))/2"
                ),
                "-r",
                f"{fps}",
                "-an",
                "-loglevel",
                "error",
                "-threads",
                "1",
                f"{dst_path / src_fname.name}",
            ],
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    main()
