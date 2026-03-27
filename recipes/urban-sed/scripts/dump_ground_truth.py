#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections import defaultdict
import csv
from pathlib import Path

import pandas as pd

from rich.progress import track


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    for split in ["train", "validate", "test"]:
        print(f"{split=}")

        ann_filename_list = list((dataset_path / "annotation" / split).glob("*.txt"))
        ann_filename_list.sort()

        groundtruth = []
        for ann_filename in track(ann_filename_list):
            wav_name = f"{ann_filename.stem}.wav"

            labels = defaultdict(lambda: [])
            with open(ann_filename) as f:
                for onset, offset, label in csv.reader(f, delimiter="\t"):
                    labels[label].append((float(onset), float(offset)))

            # merge overlapped labels
            for label, events in labels.items():
                onset, offset = events[0]
                for onset_, offset_ in events[1:]:
                    if offset >= onset_:  # merge
                        offset = max(offset, offset_)
                    else:
                        groundtruth.append((wav_name, onset, offset, label))
                        onset, offset = onset_, offset_
                groundtruth.append((wav_name, onset, offset, label))

        df_groundtruth = pd.DataFrame(groundtruth, columns=["filename", "onset", "offset", "event_label"])
        df_groundtruth.to_csv(f"{dataset_path}/annotation/{split}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
