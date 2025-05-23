#! /usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import DictConfig

from aiaccel.config import load_config, overwrite_omegaconf_dumper, print_config

# def prepare_annotations(annotation_path: Path, config: DictConfig):
#     # train
#     desed_path = Path(config.path.desed) / "metadata"
#     (annotation_path / "audioset_strong.tsv").symlink_to(desed_path / "train" / "audioset_strong.tsv")
#     (annotation_path / "synthetic21_train.tsv").symlink_to(
#         desed_path / "train" / "synthetic21_train" / "soundscapes.tsv"
#     )

#     # validation
#     (annotation_path / "validation.tsv").symlink_to(desed_path / "validation" / "validation.tsv")

#     # eval
#     public_eval_path = Path(config.path.dcase2021_public_eval) / "Ground-truth"
#     with open(public_eval_path / "mapping_file_public.tsv") as f:
#         filename_converter = {src: dst for src, dst in islice(csv.reader(f, delimiter="\t"), 1, None)}

#     with (
#         open(public_eval_path / "ground_truth_public.tsv") as fr,
#         open(annotation_path / "eval.tsv", "w") as fw,
#     ):
#         reader = csv.reader(fr, delimiter="\t")

#         writer = csv.writer(fw, delimiter="\t")
#         writer.writerow(next(reader))

#         for filename, *others in reader:
#             writer.writerow([filename_converter[filename], *others])


# def prepare_audios(audio_path: Path, annotation_path: Path, config: DictConfig):
#     # train
#     desed_path = Path(config.path.desed) / "audio"

#     audio_train_path = audio_path / "train"
#     audio_train_path.mkdir()

#     (audio_train_path / "strong_label_real").symlink_to(desed_path / "train" / "strong_label_real")
#     (audio_train_path / "synthetic21_train").symlink_to(desed_path / "train" / "synthetic21_train" / "soundscapes")

#     # validation
#     (audio_path / "validation").symlink_to(desed_path / "validation" / "validation")

#     # eval
#     with open(annotation_path / "eval.tsv") as f:
#         reader = csv.reader(f, delimiter="\t")
#         filename_list = list(set(filename for filename, *_ in islice(reader, 1, None)))

#     audio_eval_path = audio_path / "eval"
#     audio_eval_path.mkdir()
#     for filename in filename_list:
#         (audio_eval_path / filename).symlink_to(Path(config.path.dcase2021_public_eval) / "eval21" / filename)


def main():
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent

    parser = ArgumentParser()
    parser.add_argument("--audio_only", action="store_true")
    args = parser.parse_args()

    overwrite_omegaconf_dumper()

    config: DictConfig = load_config(dataset_path / "config.yaml")  # type: ignore
    print_config(config)

    audioset_path = Path(config.path.audioset)

    raw_path = dataset_path / "raw"
    raw_path.mkdir()

    (dataset_path / "raw" / "audios").symlink_to(audioset_path / "audios")
    if not args.audio_only:
        (dataset_path / "raw" / "videos").symlink_to(audioset_path / "videos")

    metadata_path = dataset_path / "metadata"
    metadata_path.mkdir()

    (metadata_path / "weak").symlink_to(audioset_path / "metadata")
    (metadata_path / "strong").symlink_to(audioset_path / "metadata_strong")

    # prepare_annotations(annotation_path, config)

    # prepare_audios(audio_path, annotation_path, config)


if __name__ == "__main__":
    main()
