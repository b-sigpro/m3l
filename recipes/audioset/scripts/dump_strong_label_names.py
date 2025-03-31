#! /usr/bin/env python3

import csv
from pathlib import Path
import pickle as pkl

from omegaconf import DictConfig

from aiaccel.utils import load_config, print_config


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    with open(dataset_path / config.strong_meta_dir / "mid_to_display_name.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        label_list = [display_name for _, display_name in reader]

    with open(dataset_path / "meta_pkl" / "label_names_strong.pkl", "wb") as f:
        pkl.dump(label_list, f)


if __name__ == "__main__":
    main()
