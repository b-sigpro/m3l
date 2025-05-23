from typing import Callable

from argparse import ArgumentParser, Namespace
import os
from pathlib import Path

from progressbar import progressbar as pbar

import torch


def separate_one(args: Namespace, unk_args: list[str], initialize: Callable, separate: Callable):
    ctx = initialize(args, unk_args)
    separate(args.src_filename, args.dst_filename, ctx, args, unk_args)


def separate_batch(args: Namespace, unk_args: list[str], initialize: Callable, separate: Callable):
    args.dst_path.mkdir(parents=True, exist_ok=True)

    src_fname_list = list(args.src_path.glob(f"*{args.src_ext}"))
    if "TASK_INDEX" in os.environ:
        start = int(os.environ["TASK_INDEX"]) - 1
        end = start + int(os.environ["TASK_STEPSIZE"])

        src_fname_list = src_fname_list[start:end]

    ctx = initialize(args, unk_args)

    for src_filename in pbar(src_fname_list):
        dst_filename = args.dst_path / src_filename.name

        separate(src_filename, dst_filename.with_suffix(args.dst_ext), ctx, args, unk_args)


@torch.inference_mode()
def main(
    add_common_args: Callable[[ArgumentParser], None],
    initialize: Callable,
    separate: Callable,
    src_ext: str,
    dst_est: str,
):
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("one")
    add_common_args(sub_parser)
    sub_parser.add_argument("src_filename", type=Path)
    sub_parser.add_argument("dst_filename", type=Path)
    sub_parser.set_defaults(handler=separate_one)

    sub_parser = sub_parsers.add_parser("batch")
    add_common_args(sub_parser)
    sub_parser.add_argument("--src_ext", type=str, default=src_ext)
    sub_parser.add_argument("--dst_ext", type=str, default=dst_est)
    sub_parser.add_argument("src_path", type=Path)
    sub_parser.add_argument("dst_path", type=Path)
    sub_parser.set_defaults(handler=separate_batch)

    args, unk_args = parser.parse_known_args()

    print("=" * 32)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 32)

    if hasattr(args, "handler"):
        args.handler(args, unk_args, initialize, separate)
    else:
        parser.print_help()
