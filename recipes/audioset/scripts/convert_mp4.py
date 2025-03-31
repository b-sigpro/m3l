#! /usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path
import subprocess

from omegaconf import DictConfig

from aiaccel.utils import load_config, print_config
from rich.progress import track


def main(args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("split", type=str, default="train")
    args = parser.parse_args()

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    fps = config.convert.fps
    image_size = config.convert.image_size

    match args.split:
        case "train":
            src_fname_list = (
                list((dataset_path / "videos" / "unbalanced_train_segments").glob("*.mp4")) + 
                list((dataset_path / "videos" / "balanced_train_segments").glob("*.mp4"))
            )
        case "eval":
            src_fname_list = list((dataset_path / "videos" / "eval_segments").glob("*.mp4"))
    src_fname_list.sort()

    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        src_fname_list = src_fname_list[start:end]

    dst_path = dataset_path / args.split / f"videos.{fps}fps"
    dst_path.mkdir(parents=True, exist_ok=True)

    # https://stackoverflow.com/questions/56628663/how-to-set-short-side-of-video-to-a-constant
    for src_fname in track(src_fname_list):
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                src_fname,
                "-vf",
                rf"scale=w=iw*min({image_size}/iw\,{image_size}/ih):h=ih*min({image_size}/iw\,{image_size}/ih), pad={image_size}:{image_size}:({image_size}-iw*min({image_size}/iw\,{image_size}/ih))/2:({image_size}-ih*min({image_size}/iw\,{image_size}/ih))/2",
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
        )


if __name__ == "__main__":
    main()
