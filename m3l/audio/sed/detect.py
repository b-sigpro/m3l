from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc

import numpy as np

import torch
from torch.nn import functional as fn  # noqa

from aiaccel.config import load_config

import librosa

from m3l.audio.sed.tasks.sed_task import SEDTask
from m3l.utils.inference import main


@dataclass
class Context:
    model: SEDTask
    config: ListConfig | DictConfig


def add_common_args(parser):
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--out_ch", type=int, default=0)
    parser.add_argument("--n_mic", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")


def initialize(args: Namespace, unk_args: list[str]):
    config = load_config(args.model_path / "merged_config.yaml")
    config = oc.merge(config, oc.from_cli(unk_args))

    checkpoint_path = args.model_path / "checkpoints" / f"{config.checkpoint_filename}.ckpt"
    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=args.device,
    )
    model.eval()

    ctx = Context(model, config)

    return ctx


def derect(src_filename: Path, dst_filename: Path, ctx: Context, args: Namespace, unk_args: list[str]):
    # load wav
    wav, sr = librosa.load(src_filename, sr=ctx.config.sample_rate, dtype=np.float32, mono=True)
    wav = torch.from_numpy(wav).to("cuda")

    # predict labels
    y_pred = ctx.model(wav.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

    with open(dst_filename, "wb") as f:
        pkl.dump({"y_pred": y_pred, "label_names": ctx.model.label_names}, f)


if __name__ == "__main__":
    main(add_common_args, initialize, derect, ".wav", ".pkl")
