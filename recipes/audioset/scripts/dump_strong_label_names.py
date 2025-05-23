#! /usr/bin/env python3

import csv
import json
from pathlib import Path


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    with open(dataset_path / "metadata" / "strong" / "mid_to_display_name.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        label_list = [display_name for _, display_name in reader]

    with open(dataset_path / "metadata" / "strong_label_names.json", "w") as f:
        json.dump(label_list, f)


if __name__ == "__main__":
    main()
