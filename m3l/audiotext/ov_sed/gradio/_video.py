# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess
import tempfile

from omegaconf import DictConfig

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np

import gradio as gr

import soundfile as sf

from m3l.audiotext.ov_sed.gradio._temporary_files import TimedTemporaryFiles


def render_plot(
    config: DictConfig,
    path: Path,
    waveform: np.ndarray,
    sample_rate: int,
    duration: float,
    logx: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
) -> list[tuple[int, int, int, int]]:
    detection_height = max(1.2, 0.22 * len(labels))

    figure = Figure(
        figsize=(config.video.width / config.video.dpi, config.video.height / config.video.dpi),
        dpi=config.video.dpi,
    )
    canvas = FigureCanvasAgg(figure)

    axes = figure.subplots(3, 1, sharex=True, gridspec_kw={"height_ratios": [1.0, 2.0, detection_height]})

    for axis in axes:
        axis.tick_params(colors=config.video.axis_color)
        axis.xaxis.label.set_color(config.video.axis_color)
        axis.yaxis.label.set_color(config.video.axis_color)
        for spine in axis.spines.values():
            spine.set_color(config.video.frame_color)

    num_waveform_points = min(waveform.size, 4000)
    indices = np.linspace(0, waveform.size - 1, num=num_waveform_points, dtype=int)

    axes[0].plot(indices / sample_rate, waveform[indices], color=config.video.waveform_color, linewidth=0.6)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("Waveform")
    axes[0].set_yticks([])
    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    axes[1].imshow(
        logx,
        origin="lower",
        aspect="auto",
        interpolation="none",
        cmap=config.video.spectrogram_cmap,
        extent=(0, duration, 0, logx.shape[0]),
    )
    axes[1].set_ylabel("Input feature")

    axes[2].imshow(
        np.flip(scores, axis=0),
        origin="lower",
        aspect="auto",
        interpolation="none",
        cmap=LinearSegmentedColormap.from_list("activation", ["#FFFFFF", config.video.activation_color]),
        vmin=0,
        vmax=1,
        extent=(0, duration, -0.5, len(labels) - 0.5),
    )
    axes[2].set_yticks(range(len(labels)), labels=labels[::-1])
    axes[2].set_ylabel("Detection result")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_xlim(0, duration)

    figure.tight_layout(pad=0.6)
    canvas.draw()

    width, height = canvas.get_width_height()

    cursor_regions = []

    for axis in axes:
        position = axis.get_position()
        cursor_regions.append(
            (
                round(position.x0 * width),
                round(position.x1 * width),
                round((1 - position.y1) * height),
                round((1 - position.y0) * height),
            )
        )

    canvas.print_png(str(path))

    return cursor_regions


def encode_video(
    config: DictConfig,
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    cursor_regions: list[tuple[int, int, int, int]],
) -> None:
    cursor_duration = max(float(np.ceil(duration * config.video.fps)) / config.video.fps, 1 / config.video.fps)
    video_duration = cursor_duration + 1 / config.video.fps

    filter_parts = []
    video_input = "[0:v]"

    for index, (left, right, top, bottom) in enumerate(cursor_regions):
        cursor_height = max(1, bottom - top)
        cursor_travel = max(1, right - left - config.video.cursor_width)
        cursor_name = f"[cursor{index}]"
        video_output = f"[v{index}]"

        filter_parts.append(
            f"color=c={config.video.cursor_color}:s={config.video.cursor_width}x{cursor_height}:"
            f"r={config.video.fps}{cursor_name}"
        )
        filter_parts.append(
            f"{video_input}{cursor_name}overlay="
            f"x='{left}+{cursor_travel}*t/{cursor_duration:.9f}':"
            f"y={top}:eval=frame{video_output}"
        )
        video_input = video_output

    filter_parts.append(f"{video_input}fps={config.video.fps},setpts=PTS-STARTPTS,format=yuv420p[v]")
    filter_complex = ";".join(filter_parts)

    command = [
        str(arg).format(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
            filter_complex=filter_complex,
            fps=config.video.fps,
            duration=f"{video_duration:.6f}",
            preset=config.video.preset,
            crf=config.video.crf,
        )
        for arg in config.video.ffmpeg_command
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise gr.Error("ffmpeg is not installed.") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or "Unknown ffmpeg error."
        raise gr.Error(f"Video encoding failed: {message}") from error


def create_video(
    config: DictConfig,
    output_files: TimedTemporaryFiles,
    waveform: np.ndarray,
    sample_rate: int,
    duration: float,
    logx: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
) -> str:
    output_path = output_files.create_path(suffix=".mp4")

    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)

        image_path = directory_path / "plot.png"
        audio_path = directory_path / "audio.wav"

        cursor_regions = render_plot(config, image_path, waveform, sample_rate, duration, logx, scores, labels)

        sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")

        encode_video(config, image_path, audio_path, output_path, duration, cursor_regions)

    return str(output_path)
