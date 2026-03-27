# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import numpy as np

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as fn

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule
import sed_scores_eval
from torchmetrics.classification.f_beta import MultilabelF1Score

from m3l.audio.common.preprocessors.preprocessor import Preprocessor


@dataclass
class Snapshot:
    logx: torch.Tensor
    y_pred: torch.Tensor
    y: torch.Tensor


class SEDTask(OptimizerLightningModule):
    """Base class for Sound Event Detection (SED) training and evaluation.

    This task defines a standard supervised training loop for sound event
    detection using strong labels. It supports segment-level and event-level
    evaluation with metrics such as F1-score and Polyphonic Sound Detection
    Score (PSDS).

    Args:
        preprocessor (Preprocessor): Preprocessing pipeline for input waveforms.
        model (nn.Module): Neural network model for SED.
        postprocessor (nn.Module): Postprocessing module applied to raw predictions.
        n_class (int): Number of sound event classes.
        label_names (list[str]): List of class label names.
        time_resolution (float): Time resolution of predictions in seconds.
        val_duration (float): Duration of validation audio clips in seconds.
        val_groundtruth (str): Path to validation ground truth annotations.
        optimizer_config (OptimizerConfig): Optimizer configuration for training.
        freezed_model (nn.Module | None, optional): Frozen model for feature
            embeddings (e.g., pretrained encoder). Defaults to None.
        threshold (float, optional): Decision threshold for event detection.
            Defaults to 0.5.
        alpha (float, optional): Weight factor for loss terms. Defaults to 1.0.
        calc_event_metrics (bool, optional): Whether to compute event-level
            metrics (PSDS, event-F1) in validation. Defaults to True.

    Attributes:
        dump (Snapshot | None): Stores a snapshot of the most recent batch
            (features, predictions, and labels) for analysis or visualization.

    Returns:
        torch.Tensor: Training loss during training step.
    """

    dump: Snapshot | None

    def __init__(
        self,
        preprocessor: Preprocessor,
        model: nn.Module,
        postprocessor: nn.Module,
        n_class: int,
        label_names: list[str],
        time_resolution: float,
        val_duration: float,
        val_groundtruth: str,
        optimizer_config: OptimizerConfig,
        freezed_model: nn.Module | None = None,
        threshold: float = 0.5,
        alpha: float = 1.0,
        calc_event_metrics: bool = True,
    ):
        super().__init__(optimizer_config)

        self.preprocessor = preprocessor
        self.model = model
        self.freezed_model = freezed_model
        self.postprocessor = postprocessor

        self.label_names = label_names
        self.time_resolution = time_resolution

        self.calc_event_metrics = calc_event_metrics

        if calc_event_metrics:
            self.val_duration = val_duration
            self.val_groundtruth = sed_scores_eval.io.read_ground_truth_events(val_groundtruth)

        self.f1_score = MultilabelF1Score(n_class, average="none")
        self.threshold = threshold

        self.alpha = alpha

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats, _ = self.preprocessor(wav)

        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)
        y_pred = self.model(feats)

        y_pred = self.postprocessor(y_pred)

        return y_pred

    def training_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.dump = None

        wav, y, _ = batch["strong"]

        # preprocess data
        feats, y = self.preprocessor(wav, y)

        # predict labels
        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)
        y_pred = self.model(feats)

        # calculate loss
        loss = fn.binary_cross_entropy(y_pred, y)

        # logging
        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                "training/loss_str": loss,
                "training/loss": loss,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        self.dump = Snapshot(
            logx=feats["logx"].detach(),
            y_pred=y_pred.detach(),
            y=y.detach(),
        )

        return loss

    def validation_step(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]],
        batch_idx: int,
    ) -> None:
        self.dump = None

        wav, y, filenames = batch["strong"]

        feats, y = self.preprocessor(wav, y)

        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)
        y_pred = self.model(feats)

        loss = fn.binary_cross_entropy(y_pred, y)

        y_pred = self.postprocessor(y_pred)
        self.update_sed_metrics(y_pred, y, filenames)

        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                "validation/loss": loss,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=wav.shape[0],
            sync_dist=True,
        )

        self.dump = Snapshot(
            logx=feats["logx"].detach(),
            y_pred=y_pred.detach(),
            y=y.detach(),
        )

    def on_validation_epoch_start(self) -> None:
        self.val_buffer = {}

    def update_sed_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        filenames: list[str],
    ) -> None:
        self.f1_score.update(y_pred, y)
        for filename, y_pred_ in zip(filenames, y_pred.detach().cpu().numpy()):
            self.val_buffer[filename] = y_pred_

    def on_validation_epoch_end(self) -> None:
        # segment f1
        f1 = self.f1_score.compute()
        self.log("validation/segment-f1", f1.mean(), prog_bar=True, on_epoch=True, sync_dist=True)
        self.f1_score.reset()

        if not self.calc_event_metrics:
            return

        # psds & event-F1
        sub_scores = {}
        for filename, y_pred_ in self.val_buffer.items():
            sub_scores[filename[:-4]] = sed_scores_eval.io.create_score_dataframe(
                y_pred_.T,
                self.time_resolution * np.arange(y_pred_.shape[-1] + 1),
                self.label_names,
            )
        scores_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(scores_list, sub_scores)
        scores = {k: v for score in scores_list for k, v in score.items()}

        durations = {k: self.val_duration for k in scores}

        if len(scores) != len(self.val_groundtruth):
            return

        psds1, *_ = sed_scores_eval.intersection_based.psds(
            scores=scores,
            ground_truth=self.val_groundtruth,
            audio_durations=durations,
            dtc_threshold=0.7,
            gtc_threshold=0.7,
            cttc_threshold=None,
            alpha_ct=0,
            alpha_st=1,
            max_efpr=100,
            num_jobs=10,
        )
        psds1 = psds1.mean()
        self.log("validation/psds1", psds1, prog_bar=True, on_epoch=True, sync_dist=True)

        psds2, *_ = sed_scores_eval.intersection_based.psds(
            scores=scores,
            ground_truth=self.val_groundtruth,
            audio_durations=durations,
            dtc_threshold=0.1,
            gtc_threshold=0.1,
            cttc_threshold=0.3,
            alpha_ct=0.5,
            alpha_st=1,
            max_efpr=100,
            num_jobs=10,
        )
        psds2 = psds2.mean()

        self.log("validation/psds2", psds2, prog_bar=True, on_epoch=True, sync_dist=True)

        event_f1, *_ = sed_scores_eval.collar_based.fscore(
            scores=scores,
            ground_truth=self.val_groundtruth,
            threshold=0.5,
            onset_collar=0.2,
            offset_collar=0.2,
            offset_collar_rate=0.2,
        )
        event_f1 = event_f1["macro_average"]

        self.log("validation/event-f1", event_f1, prog_bar=True, on_epoch=True, sync_dist=True)
