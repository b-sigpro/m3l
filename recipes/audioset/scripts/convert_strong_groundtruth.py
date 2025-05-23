#! /usr/bin/env python3

from collections import defaultdict
from pathlib import Path

import pandas as pd

from rich.progress import track
from tqdm import tqdm


def main():
    tqdm.pandas()

    # build wav_filename_dict
    wav_filename_dict = {}
    df = pd.read_csv(Path.cwd() / "metadata" / "weak" / "eval_segments.csv", skiprows=2, skipinitialspace=True)
    for _, (yt_id, start, end, _) in df.iterrows():
        wav_filename_dict[yt_id] = f"Y{yt_id}_{start:.3f}_{end:.3f}.wav"

    # build label_dict
    df = pd.read_csv(Path.cwd() / "metadata" / "strong" / "mid_to_display_name.tsv", sep="\t", names=("mid", "label"))
    label_dict = dict(zip(df["mid"], df["label"]))

    # load orig_groundtruth
    df = pd.read_csv(Path.cwd() / "metadata" / "strong" / "audioset_eval_strong.tsv", sep="\t")
    df.rename(
        columns={
            "segment_id": "filename",
            "start_time_seconds": "onset",
            "end_time_seconds": "offset",
            "label": "event_label",
        },
        inplace=True,
    )
    df["filename"] = df["filename"].str[:11].map(wav_filename_dict)
    df["event_label"] = df["event_label"].map(label_dict)
    df = df[
        df.progress_apply(lambda x: (Path.cwd() / "raw" / "audios" / "eval_segments" / x.filename).exists(), axis=1)
    ]

    orig_groundtruth = defaultdict(list)
    for _, (filename, onset, offset, label) in df.iterrows():
        orig_groundtruth[filename].append((onset, offset, label))

    # build merged_groundtruth
    merged_groundtruth = []
    for filename, event_list_ in track(orig_groundtruth.items()):
        for label in {label for *_, label in event_list_}:
            segment_list = [(st, ed) for st, ed, ll in event_list_ if ll == label]
            segment_list.sort(key=lambda x: x[0])

            target_segment = segment_list[0]
            for segment in segment_list[1:]:
                if segment[0] <= target_segment[1]:  # Check if there is overlap
                    target_segment = (target_segment[0], max(target_segment[1], segment[1]))
                else:
                    merged_groundtruth.append((filename, *target_segment, label))
                    target_segment = segment
            merged_groundtruth.append((filename, *target_segment, label))

    df = pd.DataFrame(merged_groundtruth, columns=["filename", "onset", "offset", "event_label"])
    df.to_csv(Path.cwd() / "metadata" / "strong_eval.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
