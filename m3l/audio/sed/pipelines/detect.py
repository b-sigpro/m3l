# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from argparse import ArgumentParser
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
    device: str = "cuda"

    overwrite_config: dict[str, Any] | None = None

    src_ext: str = "wav"
    dst_ext: str = "pkl"

    save_context: bool = False
    use_bf16: bool = False
    duration: float = 0.0

    def setup(self) -> None:
        self.model, self.config = load_checkpoint(
            self.model_path, "merged_config.yaml", device=self.device, overwrite_config=self.overwrite_config
        )
        self.model.eval()

    def __call__(
        self, wav: torch.Tensor, return_context: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(wav.device.type, torch.bfloat16, enabled=self.use_bf16):
            return self.model(wav, return_context=return_context)

    @torch.inference_mode()
    def process_one(self, src_filename: Path, dst_filename: Path) -> None:
        # load wav
        wav, sr = librosa.load(src_filename, sr=self.config.sample_rate, dtype=np.float32, mono=True)
        wav = torch.from_numpy(wav).to(self.device)
        assert sr == self.config.sample_rate, f"Sample rate mismatch: {sr} != {self.config.sample_rate}"

        if self.duration > 0:
            num_samples = int(self.duration * self.config.sample_rate)
            wav = wav[:num_samples]
            if wav.shape[0] < num_samples:
                wav = torch.cat((wav, wav.new_zeros(num_samples - wav.shape[0])))

        prediction = self(wav.unsqueeze(0), return_context=self.save_context)
        if self.save_context:
            y_pred, context = prediction
        else:
            y_pred = prediction

        def to_numpy(value: torch.Tensor) -> np.ndarray:
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.squeeze(0).detach().cpu().numpy()

        output = {"y_pred": to_numpy(y_pred), "label_names": self.model.label_names}
        if self.save_context:
            output["context"] = {key: to_numpy(value) for key, value in context.items()}

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
