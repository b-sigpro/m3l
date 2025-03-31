#! /usr/bin/env python3

from argparse import ArgumentParser
import json
from math import ceil, floor, isnan
from pathlib import Path

import numpy as np
import pandas as pd

from rich.progress import track


def generate_label_mat(
    filename: Path,
    annotations: pd.DataFrame,
    label_names: list[str],
    resolution: int,
):
    label_mat = np.zeros([10, 10 * 1000 // resolution], dtype=np.int8)
    for _, (_, start, end, label) in annotations.loc[annotations["filename"] == filename.name].iterrows():
        if isnan(start):
            continue

        lidx = label_names.index(label)

        start_tidx = floor(start * 1000 / resolution)
        end_tidx = ceil(end * 1000 / resolution)

        label_mat[lidx, start_tidx:end_tidx] = 1

    return label_mat


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("--resolution", type=int, default=20)
    args = parser.parse_args()

    with open(dataset_path / "metadata" / "label_names.json") as f:
        label_names: list[str] = json.load(f)

    annotations = pd.read_table(dataset_path / "annotation" / "validation.tsv")

    filename_list = list((dataset_path / "audio" / "validation").glob("*.wav"))
    filename_list.sort()

    dst_path = dataset_path / "metadata" / "label_mat" / "validation"
    dst_path.mkdir(parents=True, exist_ok=True)

    for filename in track(filename_list):
        label_mat = generate_label_mat(filename, annotations, label_names, args.resolution)

        np.save(dst_path / f"{filename.stem}.npy", label_mat)


if __name__ == "__main__":
    main()
