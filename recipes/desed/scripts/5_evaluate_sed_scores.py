#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sed_scores_eval


def main():
    parser = ArgumentParser()
    parser.add_argument("split", type=str)
    parser.add_argument("inference_path", type=Path)
    parser.add_argument("--label_resolution", type=int, default=64)
    args = parser.parse_args()

    working_directory: Path = args.inference_path / args.split

    duration_df = pd.read_csv(Path.cwd() / "metadata" / f"durations-{args.split}.csv")
    duration_dict = {
        Path(filename).stem: float(duration)
        for filename, duration in duration_df[["filename", "duration"]].itertuples(index=False, name=None)
    }

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

        durations[pkl_filename.stem] = duration_dict[pkl_filename.stem]

    # load groundtruth labels
    groundtruth = sed_scores_eval.io.read_ground_truth_events(Path.cwd() / "annotation" / f"{args.split}.tsv")
    # calculate psds1
    psds1, single_class_psds1, (etpr1, efpr1), single_class_psd_rocs1 = sed_scores_eval.intersection_based.psds(
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
    psds2, single_class_psds2, (etpr2, efpr2), _ = sed_scores_eval.intersection_based.psds(
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

    auroc, _ = sed_scores_eval.segment_based.auroc(
        scores=scores,
        ground_truth=groundtruth,
        audio_durations=durations,
        segment_length=args.label_resolution / 1000,
        num_jobs=10,
    )

    with open(working_directory / "sed_scores.pkl", "wb") as f:
        pkl.dump(
            {
                "psds1": psds1,
                "psds2": psds2,
                "event_f1": event_f1,
                "auroc": auroc["mean"],
                "single_class_psds1": single_class_psds1,
                "single_class_psds2": single_class_psds2,
                "single_class_auroc": auroc,
            },
            f,
        )

    print("PSDS1, PSDS2, Eb-F1, AUROC")
    print(f"{psds1:.3f}, {psds2:.3f}, {event_f1['macro_average']:.3f}, {auroc['mean']:.3f}")

    pprint.pprint(single_class_psds1)

    fig, ax = plt.subplots()
    for cls, (tpr, efpr, _) in single_class_psd_rocs1.items():
        ax.step(efpr, tpr, where="post", label=str(cls))

    ax.set_xlabel("eFPR (/unit time)")
    ax.set_ylabel("eTPR")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(ncol=1)
    ax.set_title("Single-class PSD-ROC")

    fig.tight_layout()
    fig.savefig(f"{working_directory}/psd_roc_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(6, 4))
    plt.step(efpr1, etpr1, where="post", label=f"PSDS1 (DTC=0.7,GTC=0.7) = {psds1:.3f}")
    plt.step(efpr2, etpr2, where="post", label=f"PSDS2 (DTC=0.1,GTC=0.1) = {psds2:.3f}")
    plt.xlabel("eFPR (/hour)")
    plt.ylabel("eTPR")
    plt.title("PSD-ROC under PSDS1/PSDS2 conditions")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{working_directory}/psd_roc_psds1_2.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
