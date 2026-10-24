# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import pickle as pkl

from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc

import numpy as np

import torch
from torch.nn import functional as fn  # noqa

from aiaccel.torch.lightning import load_checkpoint

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
    parser.add_argument("--save-context", action="store_true")


def initialize(args: Namespace, unk_args: list[str]):
    model, config = load_checkpoint(args.model_path, device=args.device, overwrite_config=oc.from_cli(unk_args))
    model.eval()

    ctx = Context(model, config)

    return ctx


def derect(src_filename: Path, dst_filename: Path, ctx: Context, args: Namespace, unk_args: list[str]):
    # load wav
    wav, sr = librosa.load(src_filename, sr=ctx.config.sample_rate, dtype=np.float32, mono=True)
    wav = torch.from_numpy(wav).to("cuda")

    # predict labels
    prediction = ctx.model(wav.unsqueeze(0), return_context=args.save_context)
    if args.save_context:
        y_pred, context = prediction
    else:
        y_pred = prediction

    output = {"y_pred": y_pred.squeeze(0).detach().cpu().numpy(), "label_names": ctx.model.label_names}
    if args.save_context:
        output["context"] = {key: value.squeeze(0).detach().cpu().numpy() for key, value in context.items()}

    with open(dst_filename, "wb") as f:
        pkl.dump(output, f)


if __name__ == "__main__":
    main(add_common_args, initialize, derect, ".wav", ".pkl")
