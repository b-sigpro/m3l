# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as fn

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule
from torchmetrics.classification import MultilabelAveragePrecision

from m3l.audio.common.preprocessors.preprocessor import Preprocessor


@dataclass
class Snapshot:
    logx: torch.Tensor
    y_pred: torch.Tensor
    y: torch.Tensor


class TaggingTask(OptimizerLightningModule):
    """Lightning module for multilabel audio tagging.

    This task handles preprocessing, model inference, training, and
    evaluation for multilabel classification problems.

    Args:
        preprocessor (Preprocessor): Module to convert raw waveforms
            into model input features.
        model (nn.Module): Neural network for audio tagging.
        n_class (int): Number of target classes.
        optimizer_config (OptimizerConfig): Optimizer and scheduler
            configuration for Lightning.
        label_smoothing (float, optional): Label smoothing factor for
            binary cross-entropy loss. Defaults to ``0.0``.

    Attributes:
        dump (Snapshot | None): Stores the most recent batch for
            visualization or debugging.
        avg_precision (MultilabelAveragePrecision): Metric for evaluating
            mean average precision on validation data.

    Shape:
        - Input: ``wav`` of shape ``(B, T)`` containing audio waveforms.
        - Output: ``torch.Tensor`` of shape ``(B, n_class)`` with
          predicted probabilities or logits.
    """

    dump: Snapshot | None

    def __init__(
        self,
        preprocessor: Preprocessor,
        model: nn.Module,
        n_class: int,
        optimizer_config: OptimizerConfig,
        label_smoothing: float = 0.0,
    ):
        super().__init__(optimizer_config)

        self.preprocessor = preprocessor
        self.model = model

        self.label_smoothing = label_smoothing

        self.avg_precision = MultilabelAveragePrecision(n_class)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats, _ = self.preprocessor(wav)

        y_pred = self.model(feats)

        return y_pred

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[str]],
        batch_idx: int,
        log_prefix: str = "training",
    ) -> torch.Tensor:
        self.dump = None

        wav, y, _ = batch

        # preprocess data
        with torch.autocast("cuda", enabled=False):
            feats, y = self.preprocessor(wav, y)

        # predict labels
        y_pred = self.model(feats, return_logits=True)

        # calculate loss
        y_ = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing if self.label_smoothing > 0 else y
        loss = fn.binary_cross_entropy_with_logits(y_pred, y_)

        # logging
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

        if log_prefix == "validation":
            self.avg_precision.update(y_pred.sigmoid(), y.int())
            self.log_dict(
                {
                    "validation/mean_average_precision": self.avg_precision,
                },
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=wav.shape[0],
                sync_dist=True,
            )

        self.dump = Snapshot(
            logx=feats["logx"][0].detach(),
            y_pred=y_pred[0].detach(),
            y=y[0].detach(),
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, list[str]],
        batch_idx: int,
    ) -> None:
        self.training_step(batch, batch_idx, log_prefix="validation")
