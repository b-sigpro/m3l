#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl


def main():
    parser = ArgumentParser()
    parser.add_argument("inference_path", type=Path)
    args = parser.parse_args()

    score_str_list = []
    for split in ["validation", "eval"]:
        with open(args.inference_path / split / "sed_scores.pkl", "rb") as f:
            sed_scores = pkl.load(f)
            # pkl.dump({
            #     "psds1": psds1,
            #     "psds2": psds2,
            #     "event_f1": event_f1,
            # }, f)

        psds1 = sed_scores["psds1"]
        psds2 = sed_scores["psds2"]
        event_f1 = sed_scores["event_f1"]

        score_str = f"{psds1:.3f}, {psds2:.3f}, {event_f1['micro_average']:.3f}"
        score_str_list.append(score_str)

    print(", ".join(score_str_list))


if __name__ == "__main__":
    main()
