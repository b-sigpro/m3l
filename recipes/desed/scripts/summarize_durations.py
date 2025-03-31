#! /usr/bin/env python3

from pathlib import Path

import pandas as pd

from rich.progress import track

import soundfile as sf


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    for split in ["validation", "eval"]:
        print(f"{split=}")

        filename_list = list((dataset_path / "audio" / split).glob("*.wav"))
        filename_list.sort()

        metadata = []
        for filename in track(filename_list):
            duration = sf.info(filename).duration
            metadata.append((filename.name, duration))

        df_metadata = pd.DataFrame(metadata, columns=["filename", "duration"])
        df_metadata.to_csv(dataset_path / "metadata" / f"durations-{split}.csv", index=False)


if __name__ == "__main__":
    main()
