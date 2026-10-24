# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import DictConfig

from einops import reduce
import numpy as np

import torch

from aiaccel.config import prepare_config, print_config
from aiaccel.torch.lightning import load_checkpoint
import gradio as gr

import librosa

from m3l.audiotext.ov_sed.gradio._spaces import with_spaces_gpu
from m3l.audiotext.ov_sed.gradio._temporary_files import TimedTemporaryFiles
from m3l.audiotext.ov_sed.gradio._video import create_video


def to_mono(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform)

    if np.issubdtype(waveform.dtype, np.signedinteger):
        waveform = waveform.astype(np.float32) / -np.iinfo(waveform.dtype).min
    else:
        waveform = waveform.astype(np.float32)

    if waveform.ndim == 2:
        waveform = reduce(waveform, "time channel -> time", "mean")

    if waveform.ndim != 1:
        raise ValueError(f"Unexpected audio shape: {waveform.shape}")

    return waveform


def prepare_audio(audio: tuple[int, np.ndarray], sample_rate: int, max_duration: float) -> tuple[torch.Tensor, float]:
    input_sample_rate, waveform = audio
    waveform = to_mono(waveform)

    if waveform.size == 0:
        raise gr.Error("Audio is empty.")

    if not np.isfinite(waveform).all():
        raise gr.Error("Audio contains non-finite values.")

    waveform = waveform[: round(max_duration * input_sample_rate)]

    if input_sample_rate != sample_rate:
        waveform = librosa.resample(y=waveform, orig_sr=input_sample_rate, target_sr=sample_rate)

    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = waveform[-round(max_duration * sample_rate) :]
    duration = waveform.size / sample_rate

    return torch.from_numpy(waveform), duration


def parse_labels(text: str, max_labels: int) -> list[str]:
    labels = [line.strip() for line in text.splitlines() if line.strip()]

    if not labels:
        raise gr.Error("Enter at least one detection label.")

    if len(labels) > max_labels:
        raise gr.Error(f"Enter at most {max_labels} labels.")

    return labels


class OpenVocabularySED:
    def __init__(
        self,
        model_path: str,
        config_path: Path = Path(__file__).parent / "config/default.yaml",
        device: str = "auto",
        strict: bool | None = None,
        weights_only: bool | None = None,
    ):
        self.config: DictConfig = prepare_config(config_path)
        print_config(self.config)

        # device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # load model
        model, config = load_checkpoint(
            model_path,
            device=self.device,
            strict=strict,
            weights_only=weights_only,
        )
        model.to(self.device).eval()

        self.model = model
        self.sample_rate = int(config.sample_rate)

        self.output_files = TimedTemporaryFiles(ttl=self.config.output.ttl)

    def prepare_callback(
        self, audio: tuple[int, np.ndarray] | None, label_text: str
    ) -> tuple[np.ndarray, float, list[str]]:
        if audio is None:
            raise gr.Error("Upload or record audio first.")

        labels = parse_labels(label_text, self.config.audio.max_labels)
        waveform, duration = prepare_audio(audio, self.sample_rate, self.config.audio.max_duration)
        return waveform.numpy(), duration, labels

    @with_spaces_gpu
    def predict_callback(
        self, prepared: tuple[np.ndarray, float, list[str]]
    ) -> tuple[np.ndarray, float, list[str], np.ndarray, np.ndarray]:
        waveform, duration, labels = prepared

        with torch.inference_mode():
            model_input = torch.from_numpy(waveform).unsqueeze(0).to(self.device)

            prediction, context = self.model(model_input, [labels], return_context=True)
            logx = context["logx"].detach().float().cpu().numpy().squeeze(0)
            if logx.ndim != 2:
                raise ValueError(f"Unexpected logx shape: {logx.shape}")

            scores = prediction.squeeze(0).detach().float().cpu().numpy()
            assert scores.shape[0] == len(labels), f"Expected {len(labels)} labels, received {scores.shape[0]}."

        return waveform, duration, labels, logx, scores

    def create_video_callback(self, result: tuple[np.ndarray, float, list[str], np.ndarray, np.ndarray]) -> str:
        waveform, duration, labels, logx, scores = result
        return create_video(
            self.config,
            self.output_files,
            waveform,
            self.sample_rate,
            duration,
            logx,
            scores,
            labels,
        )

    def create_demo(self) -> gr.Blocks:
        config = self.config
        description = config.app.description

        with gr.Blocks(title=config.app.title, delete_cache=(3600, 3600)) as demo:
            audio_state = gr.State(None)
            prepared_state = gr.State(None)
            result_state = gr.State(None)

            gr.Markdown(f"# {config.app.title}\n\n{description}")

            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    input_audio = gr.Audio(label="Input audio", sources=["upload", "microphone"], type="numpy")
                    gr.on(
                        triggers=[input_audio.upload, input_audio.stop_recording],
                        fn=lambda audio: audio,
                        inputs=input_audio,
                        outputs=audio_state,
                        queue=False,
                    )

                    label_text = gr.Textbox(
                        label="Target labels",
                        value="\n".join(config.labels),
                        lines=len(config.labels),
                    )

                    detect = gr.Button("Detect", variant="primary")

                with gr.Column(scale=3, min_width=480):
                    output_video = gr.Video(label="Detection result", format="mp4", interactive=False)

            (
                detect.click(self.prepare_callback, inputs=[audio_state, label_text], outputs=prepared_state)
                .success(self.predict_callback, inputs=prepared_state, outputs=result_state)
                .success(self.create_video_callback, inputs=result_state, outputs=output_video)
            )

        return demo


if __name__ == "__main__":
    parser = ArgumentParser(description="Open-vocabulary sound event detection demo")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--disable_strict", action="store_false", default=True)
    parser.add_argument("--disable_weights_only", action="store_false", default=True)
    args = parser.parse_args()

    app = OpenVocabularySED(
        args.model_path,
        args.config,
        args.device,
        strict=args.disable_strict,
        weights_only=args.disable_weights_only,
    )
    app.create_demo().queue(default_concurrency_limit=1).launch(show_error=True)
