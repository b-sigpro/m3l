#!/usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections import defaultdict
import csv
from itertools import islice
import json
from pathlib import Path

from rich.progress import track


def main():
    dataset_path = Path(__file__).parent.parent

    # use prepare_gloqbals()
    mid_idx_dict = {}
    with open(dataset_path / "metadata" / "strong" / "mid_to_display_name.tsv", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        mid_idx_dict = {mid: idx for idx, (mid, name) in enumerate(reader)}

    name_resolver = {}
    for split_name in ["balanced_train_segments", "unbalanced_train_segments"]:
        print(f"{split_name}")
        with open(dataset_path / "metadata" / "weak" / f"{split_name}.csv") as f:
            reader = csv.reader(f, skipinitialspace=True)
            for ytid, start, end, _ in track(islice(reader, 3, None)):
                name = f"Y{ytid}_{float(start):.3f}_{float(end):.3f}.wav"

                name_resolver[ytid] = name

    set_metadata = defaultdict(set)
    with open(dataset_path / "metadata" / "strong" / "audioset_train_strong.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        for segment_id, _, _, label in track(islice(reader, 1, None)):
            ytid, _ = segment_id.rsplit("_", 1)

            set_metadata[name_resolver[ytid]].add(mid_idx_dict[label])

    metadata = {name: sorted(list(lidx)) for name, lidx in set_metadata.items()}

    # save dir
    with open(dataset_path / "metadata" / "strong_train.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
