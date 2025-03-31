# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from functools import partial

import torch
from torch import nn

from torchaudio.transforms import AmplitudeToDB, MelSpectrogram

from m3l.common.nn import TupleSequential


class Preprocessor(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        wav_transforms: list[nn.Module] | None = None,
        mel_transforms: list[nn.Module] | None = None,
        logmel_transforms: list[nn.Module] | None = None,
    ):
        super().__init__()

        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=partial(torch.hamming_window, periodic=False),
            power=1,
        )

        self.to_db = AmplitudeToDB(stype="amplitude")
        self.to_db.amin = 1e-5

        self.wav_transforms = nn.Sequential(*wav_transforms if wav_transforms is not None else [])
        self.mel_transforms = TupleSequential(*mel_transforms if mel_transforms is not None else [])
        self.logmel_transforms = TupleSequential(*logmel_transforms if logmel_transforms is not None else [])

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, wav: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        B, _ = wav.shape

        wav = self.wav_transforms(wav)

        x, y = self.mel_transforms(self.mel(wav), y)

        logx, y = self.logmel_transforms(self.to_db(x).clamp(-50, 80), y)

        return {"wav": wav, "x": x, "logx": logx}, y
