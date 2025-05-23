#! /usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl

import numpy as np

import sed_scores_eval


def main():
    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("inference_path", type=Path)
    parser.add_argument("--label_resolution", type=int, default=64)
    args = parser.parse_args()

    working_directory: Path = args.inference_path / args.split

    # load predicted labels
    scores, durations = {}, {}
    for pkl_filename in (working_directory / "scores").glob("*.pkl"):
        with open(pkl_filename, "rb") as f:
            data = pkl.load(f)

        y_pred = data["y_pred"]
        label_names = data["label_names"]

        scores[pkl_filename.stem] = sed_scores_eval.io.create_score_dataframe(
            y_pred.T,
            args.label_resolution / 1000 * np.arange(y_pred.shape[-1] + 1),
            label_names,
        )

        durations[pkl_filename.stem] = 10.0

    # load groundtruth labels
    groundtruth = sed_scores_eval.io.read_ground_truth_events(Path.cwd() / "annotation" / f"{args.split}.tsv")

    # calculate psds1
    psds1, *_ = sed_scores_eval.intersection_based.psds(
        scores=scores,
        ground_truth=groundtruth,
        audio_durations=durations,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0,
        alpha_st=1,
        max_efpr=100,
        num_jobs=10,
    )
    psds1 = psds1.mean()

    # calculate psds2
    psds2, *_ = sed_scores_eval.intersection_based.psds(
        scores=scores,
        ground_truth=groundtruth,
        audio_durations=durations,
        dtc_threshold=0.1,
        gtc_threshold=0.1,
        cttc_threshold=0.3,
        alpha_ct=0.5,
        alpha_st=1,
        max_efpr=100,
        num_jobs=10,
    )
    psds2 = psds2.mean()

    # calculate event F1
    event_f1, *_ = sed_scores_eval.collar_based.fscore(
        scores=scores,
        ground_truth=groundtruth,
        threshold=0.5,
        onset_collar=0.2,
        offset_collar=0.2,
        offset_collar_rate=0.2,
    )

    with open(working_directory / "sed_scores.pkl", "wb") as f:
        pkl.dump(
            {
                "psds1": psds1,
                "psds2": psds2,
                "event_f1": event_f1,
            },
            f,
        )

    print("PSDS1, PSDS2, Eb-F1")
    print(f"{psds1:.3f}, {psds2:.3f}, {event_f1['micro_average']:.3f}")


if __name__ == "__main__":
    main()
