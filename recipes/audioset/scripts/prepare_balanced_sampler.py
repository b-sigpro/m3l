#!/usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import csv
from itertools import islice
import json
from pathlib import Path

from rich.progress import track


def main():
    dataset_path = Path(__file__).parent.parent

    # use prepare_globals()
    mid_idx_dict = {}
    with open(dataset_path / "metadata" / "weak" / "class_labels_indices.csv", newline="") as f:
        reader = csv.DictReader(f)
        mid_idx_dict = {row["mid"].strip(): int(row["index"].strip()) for row in reader}

    label_info_list = []
    for split_name in ["balanced_train_segments", "unbalanced_train_segments"]:
        print(f"{split_name}")
        with open(dataset_path / "metadata" / "weak" / f"{split_name}.csv") as f:
            reader = csv.reader(f, skipinitialspace=True)
            for yt_id, start, end, labels in track(islice(reader, 3, None)):
                name = f"Y{yt_id}_{float(start):.3f}_{float(end):.3f}.wav"

                cidx = [mid_idx_dict[lbl] for lbl in labels.split(",")]
                label_info_list.append((name, cidx))

    metadata = {name: sorted(list(set(cidx))) for name, cidx in label_info_list}

    # save dir
    with open(dataset_path / "metadata" / "weak_train.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
