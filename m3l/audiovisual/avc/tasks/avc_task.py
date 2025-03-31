# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as fn  # noqa

# from ci_sdr.pt import ci_sdr_loss
from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


@dataclass
class DumpData:
    logx: torch.Tensor
    frames: torch.Tensor
    labels: torch.Tensor
    za: torch.Tensor
    zv: torch.Tensor


class AVCTask(OptimizerLightningModule):
    def __init__(
        self,
        audio_preprocessor: nn.Module,
        audio_encoder: nn.Module,
        video_encoder: nn.Module,
        optimizer_config: OptimizerConfig,
        tau: float = 0.1,
    ):
        super().__init__(optimizer_config)

        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.audio_preprocessor = audio_preprocessor

        self.tau = tau

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        log_prefix: str = "training",
    ):
        self.dump = None

        wav, frames = batch

        # preprocess audio data
        logx = self.audio_preprocessor(wav)

        # predict embeddings
        za = self.audio_encoder(logx).mean(dim=-1)
        zv = self.video_encoder(frames).mean(dim=(-1, -2))

        # calculate InfoNCELoss (cosine_sim)
        za = fn.normalize(za, dim=-1)
        zv = fn.normalize(zv, dim=-1)

        loss = 0.0
        sim_v = zv @ za.transpose(-2, -1)  # [B, B]
        sim_a = za @ zv.transpose(-2, -1)  # [B, B]

        for i in range(len(sim_v)):
            sim_v[i] = sim_v[i].roll(shifts=-i)
            sim_a[i] = sim_a[i].roll(shifts=-i)

        sim_a = sim_a[:, : self.n_negatives + 1]  # [B, self.n_negatives+1]
        sim_v = sim_v[:, : self.n_negatives + 1]  # [B, self.n_negatives+1]

        labels = torch.zeros(zv.shape[0], device=zv.device, dtype=torch.long)
        loss = (fn.cross_entropy(sim_v / self.tau, labels) + fn.cross_entropy(sim_a / self.tau, labels)) / 2

        self.dump = DumpData(
            logx=logx[0].detach(),
            frames=frames[0].detach(),
            labels=labels[0].detach(),
            za=za[0].detach(),
            zv=zv[0].detach(),
        )

        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                f"{log_prefix}/loss": loss,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=wav.shape[0],
            sync_dist=True,
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        return self.training_step(batch, batch_idx, log_prefix="validation")
