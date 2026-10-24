# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
from types import SimpleNamespace

from omegaconf import OmegaConf

import numpy as np

import pytest

from m3l.audiotext.ov_sed.gradio import _spaces
from m3l.audiotext.ov_sed.gradio._temporary_files import TimedTemporaryFiles
from m3l.audiotext.ov_sed.gradio._video import create_video, render_plot


@pytest.fixture
def demo_config():
    config_path = Path(__file__).resolve().parents[1] / "m3l/audiotext/ov_sed/gradio/config/default.yaml"
    config = OmegaConf.load(config_path)
    config.video.width = 640
    config.video.height = 360
    config.video.fps = 5
    return config


def test_render_plot(demo_config, tmp_path):
    waveform = np.sin(np.linspace(0, 40, 8000)).astype(np.float32)
    logx = np.linspace(-50, 10, 640, dtype=np.float32).reshape(32, 20)
    scores = np.linspace(0, 1, 40, dtype=np.float32).reshape(2, 20)
    output = tmp_path / "plot.png"

    regions = render_plot(demo_config, output, waveform, 8000, 1.0, logx, scores, ["cat", "dog"])

    assert output.is_file() and output.stat().st_size > 0
    assert len(regions) == 3
    assert all(0 <= left < right <= 640 and 0 <= top < bottom <= 360 for left, right, top, bottom in regions)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_create_video(demo_config):
    sample_rate = 8000
    waveform = (0.2 * np.sin(np.linspace(0, 50, sample_rate // 2))).astype(np.float32)
    logx = np.ones((32, 20), dtype=np.float32)
    scores = np.linspace(0, 1, 40, dtype=np.float32).reshape(2, 20)
    output_files = TimedTemporaryFiles(ttl=3600)

    path = Path(create_video(demo_config, output_files, waveform, sample_rate, 0.5, logx, scores, ["cat", "dog"]))

    assert path.is_file() and path.suffix == ".mp4" and path.stat().st_size > 1000
    assert path.parent == output_files.directory


def test_spaces_gpu_wrapper(monkeypatch):
    def identity():
        return 123

    monkeypatch.setattr(_spaces, "spaces", None)
    assert _spaces.with_spaces_gpu(identity) is identity

    called = []

    def gpu(duration):
        called.append(duration)
        return lambda fn: lambda: fn() + 1

    monkeypatch.setattr(_spaces, "spaces", SimpleNamespace(GPU=gpu))
    assert _spaces.with_spaces_gpu(identity)() == 124
    assert called == [30]
