#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import csv
from itertools import islice
import json
from pathlib import Path


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    with open(dataset_path / "metadata" / "weak" / "class_labels_indices.csv") as f:
        reader = csv.reader(f)
        label_list = [display_name for _, _, display_name in islice(reader, 1, None)]

    with open(dataset_path / "metadata" / "weak_label_names.json", "w") as f:
        json.dump(label_list, f)


if __name__ == "__main__":
    main()
