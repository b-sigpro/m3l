#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl


def main():
    parser = ArgumentParser()
    parser.add_argument("inference_path", type=Path)
    args = parser.parse_args()

    score_str_list = []
    for split in ["validate", "test"]:
        with open(args.inference_path / split / "sed_scores.pkl", "rb") as f:
            sed_scores = pkl.load(f)

        psds1 = sed_scores["psds1"]
        psds2 = sed_scores["psds2"]
        event_f1 = sed_scores["event_f1"]

        score_str = f"{psds1:.3f}, {psds2:.3f}, {event_f1['macro_average']:.3f}"
        score_str_list.append(score_str)

    print(", ".join(score_str_list))


if __name__ == "__main__":
    main()
