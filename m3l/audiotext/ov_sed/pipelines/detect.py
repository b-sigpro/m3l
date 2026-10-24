# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from argparse import ArgumentParser
import json
from pathlib import Path
import pickle as pkl

from omegaconf import OmegaConf as oc  # noqa: N813

import numpy as np

import torch

from aiaccel.torch.lightning import load_checkpoint
from aiaccel.torch.pipelines import BasePipeline, reorder_fields
import attrs

import librosa


@attrs.define(slots=False, field_transformer=reorder_fields)
class DetectPipeline(BasePipeline):
    model_path: Path
    label_names_filename: Path

    label_override_filename: Path = None
    save_context: bool = False

    device: str = "cuda"

    overwrite_config: dict[str, Any] | None = None

    src_ext: str = "wav"
    dst_ext: str = "pkl"

    def setup(self) -> None:
        self.model, self.config = load_checkpoint(
            self.model_path,
            "merged_config.yaml",
            device=self.device,
            overwrite_config=self.overwrite_config,
        )
        self.model.eval()

        with self.label_names_filename.open("r", encoding="utf-8") as f:
            self.label_names = json.load(f)

        self.model_labels = self.label_names
        if self.label_override_filename is not None:
            with self.label_override_filename.open("r", encoding="utf-8") as f:
                self.model_labels = json.load(f)

    def __call__(
        self, wav: torch.Tensor, return_context: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.model(wav, [self.model_labels], return_context=return_context)

    @torch.inference_mode()
    def process_one(self, src_filename: Path, dst_filename: Path) -> None:
        # load wav
        wav, sr = librosa.load(src_filename, sr=self.config.sample_rate, dtype=np.float32, mono=True)
        wav = torch.from_numpy(wav).to(self.device)
        assert sr == self.config.sample_rate, f"Sample rate mismatch: {sr} != {self.config.sample_rate}"

        prediction = self(wav.unsqueeze(0), return_context=self.save_context)
        if self.save_context:
            y_pred, context = prediction
        else:
            y_pred = prediction

        output = {"y_pred": y_pred.squeeze(0).detach().cpu().numpy(), "label_names": self.label_names}
        if self.save_context:
            output["context"] = {key: value.squeeze(0).detach().cpu().numpy() for key, value in context.items()}

        with open(dst_filename, "wb") as f:
            pkl.dump(output, f)

    @classmethod
    def _prepare_parser(cls, fields: list[attrs.Attribute]) -> ArgumentParser:
        return super()._prepare_parser(list(filter(lambda f: f.name != "overwrite_config", fields)))

    @classmethod
    def _process_unk_args(cls, unk_args: list[str], kwargs: dict[str, Any], parser: ArgumentParser) -> dict[str, Any]:
        return kwargs | {"overwrite_config": oc.from_cli(unk_args)}


if __name__ == "__main__":
    DetectPipeline.main()
