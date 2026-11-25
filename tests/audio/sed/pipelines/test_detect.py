# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import pickle
import sys
from unittest.mock import Mock

from omegaconf import OmegaConf

import numpy as np

import torch

import pytest

import soundfile as sf

from m3l.audio.sed.pipelines import detect


class DummyModel(torch.nn.Module):
    label_names = ["cat", "dog"]

    def forward(self, wav, return_context=False):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        scores = wav.unsqueeze(1).repeat(1, 2, 1)
        if torch.is_autocast_enabled("cpu"):
            scores = scores.to(torch.bfloat16)
        if return_context:
            return scores, {"waveform": scores[:, 0], "indices": torch.arange(wav.shape[-1]).unsqueeze(0)}
        return scores


@pytest.mark.parametrize(
    ("mode", "duration", "save_context", "use_bf16"),
    [
        ("one", 0.0, False, False),
        ("batch", 0.0, True, False),
        ("one", 0.05, True, False),
        ("batch", 0.2, True, True),
    ],
)
def test_detect_pipeline(monkeypatch, tmp_path, mode, duration, save_context, use_bf16):
    model_path = tmp_path / "model"
    sample_rate = 8000
    waveform = np.full(800, 0.25, dtype=np.float32)
    src_path = tmp_path / "audio"
    src_path.mkdir()
    names = ["first", "second"] if mode == "batch" else ["first"]
    for name in names:
        sf.write(src_path / f"{name}.wav", waveform, sample_rate, subtype="FLOAT")

    loader = Mock(return_value=(DummyModel(), OmegaConf.create({"sample_rate": sample_rate})))
    monkeypatch.setattr(detect, "load_checkpoint", loader)
    monkeypatch.setattr("aiaccel.torch.pipelines.base_pipeline.split_tasks", lambda files: files)

    dst_path = tmp_path / "scores"
    if mode == "one":
        src, dst = src_path / "first.wav", dst_path / "first.pkl"
    else:
        src, dst = src_path, dst_path

    args = [
        "detect",
        mode,
        str(model_path),
        str(src),
        str(dst),
        "--device=cpu",
        f"--duration={duration}",
        "sample_rate=8000",
    ]
    if save_context:
        args.append("--save_context")
    if use_bf16:
        args.append("--use_bf16")
    monkeypatch.setattr(sys, "argv", args)

    detect.DetectPipeline.main()

    loader.assert_called_once()
    assert loader.call_args.args == (model_path, "merged_config.yaml")
    assert loader.call_args.kwargs["device"] == "cpu"
    assert loader.call_args.kwargs["overwrite_config"].sample_rate == sample_rate

    expected = waveform
    if duration > 0:
        size = int(duration * sample_rate)
        expected = np.pad(waveform[:size], (0, max(0, size - len(waveform))))

    assert sorted(p.stem for p in dst_path.glob("*.pkl")) == names
    for name in names:
        with (dst_path / f"{name}.pkl").open("rb") as f:
            output = pickle.load(f)

        assert output["label_names"] == DummyModel.label_names
        assert output["y_pred"].dtype == np.float32
        np.testing.assert_array_equal(output["y_pred"], np.stack([expected, expected]))
        assert ("context" in output) == save_context
        if save_context:
            np.testing.assert_array_equal(output["context"]["waveform"], expected)
            assert output["context"]["waveform"].dtype == np.float32
            np.testing.assert_array_equal(output["context"]["indices"], np.arange(len(expected)))
            assert output["context"]["indices"].dtype == np.int64
