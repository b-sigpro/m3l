#! /usr/bin/env python
# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import DictConfig

from aiaccel.config import prepare_config


def prepare_annotations(annotation_path: Path, config: DictConfig):
    urban_sed_path = Path(config.path.urban_sed) / "annotations"

    for split in ["train", "validate", "test"]:
        (annotation_path / split).mkdir(exist_ok=True)

        for filename in (urban_sed_path / split).glob("*.txt"):
            if filename.stem.startswith("."):
                continue

            (annotation_path / split / filename.name).symlink_to(filename)


def prepare_audios(audio_path: Path, config: DictConfig):
    urban_sed_path = Path(config.path.urban_sed) / "audio"

    for split in ["train", "validate", "test"]:
        (audio_path / split).mkdir(exist_ok=True)

        for filename in (urban_sed_path / split).glob("*.wav"):
            if filename.stem.startswith("."):
                continue

            (audio_path / split / filename.name).symlink_to(filename)


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    config = prepare_config(
        dataset_path / "config.yaml",
        print_config=True,
    )

    annotation_path = dataset_path / "annotation"
    annotation_path.mkdir(exist_ok=True, parents=True)

    audio_path = dataset_path / "audio"
    audio_path.mkdir(exist_ok=True, parents=True)

    metadata_path = dataset_path / "metadata"
    metadata_path.mkdir(exist_ok=True, parents=True)

    prepare_annotations(annotation_path, config)

    prepare_audios(audio_path, config)


if __name__ == "__main__":
    main()
