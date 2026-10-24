# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from dataclasses import dataclass

from einops import rearrange, repeat

import torch
from torch import nn
from torch.nn import functional as fn

from aiaccel.torch.lightning import OptimizerConfig

from m3l.audio.common.preprocessors.preprocessor import Preprocessor
from m3l.audio.sed.tasks.sed_task import SEDTask


@dataclass
class Snapshot:
    logx: torch.Tensor
    y_pred: torch.Tensor
    y: torch.Tensor
    labels: list[str]


class DOOMSEDTask(SEDTask):
    dump: Snapshot | None

    def __init__(
        self,
        preprocessor: Preprocessor,
        model: nn.Module,
        postprocessor: nn.Module,
        time_resolution: float,
        optimizer_config: OptimizerConfig,
        freezed_model: nn.Module | None = None,
        alpha_clip: float = 1.0,
        calcurate_all_block: bool = False,
        n_block: int = 4,
    ):
        super().__init__(
            preprocessor=preprocessor,
            model=model,
            postprocessor=postprocessor,
            n_class=1,
            label_names=["dummy"],
            time_resolution=time_resolution,
            optimizer_config=optimizer_config,
            freezed_model=freezed_model,
            calc_event_metrics=False,
        )

        self.alpha_clip = alpha_clip
        self.calcurate_all_block = calcurate_all_block
        self.n_block = n_block

        # self.text_normalizer = text_normalizer

    def forward(
        self, wav: torch.Tensor, labels: list[list[str]], return_context: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:  # type: ignore[override]
        feats, _ = self.preprocessor(wav)

        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)

        y_pred = self.model(feats, labels)

        y_pred = self.postprocessor(y_pred)

        if return_context:
            return y_pred, feats
        return y_pred

    def training_step(  # type: ignore[override]
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, list[str], list[str]]],
        batch_idx: int,
    ) -> torch.Tensor:
        self.dump = None

        wav, y, labels, _ = batch["strong"]
        labels = [list(label) for label in zip(*labels)]

        # labels = self.text_normalizer(labels)

        # preprocess data
        feats, y = self.preprocessor(wav, y)

        # predict labels
        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)

        if self.calcurate_all_block:
            y_pred_str_logit, y_pred_wek_logit = self.model(
                feats, labels, return_logits=True, return_weak=True, return_all=True
            )
            y = repeat(y, "b ... -> (block b) ...", block=self.n_block)
            y_pred_str_logit = rearrange(y_pred_str_logit, "n b ... -> (n b) ...")
            y_pred_wek_logit = rearrange(y_pred_wek_logit, "n b ... -> (n b) ...")
        else:
            y_pred_str_logit, y_pred_wek_logit = self.model(feats, labels, return_logits=True, return_weak=True)

        # make mask
        loss_str = fn.binary_cross_entropy_with_logits(y_pred_str_logit, y)
        loss_str_clip = fn.binary_cross_entropy_with_logits(y_pred_wek_logit, torch.amax(y, dim=-1))

        loss = loss_str + loss_str_clip * self.alpha_clip

        # logging
        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                "training/loss_str": loss_str,
                "training/loss_str_clip": loss_str_clip,
                "training/loss": loss,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        self.dump = Snapshot(
            logx=feats["logx"][-1].detach(),
            y_pred=y_pred_str_logit[-1].sigmoid().detach(),
            y=y[-1].detach(),
            labels=labels[-1],
        )

        return loss

    def validation_step(  # type: ignore[override]
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, list[str], list[str]]],
        batch_idx: int,
    ) -> None:
        self.dump = None

        wav, y, labels, filenames = batch["strong"]
        labels = [list(label) for label in zip(*labels)]

        # labels = self.text_normalizer(labels)

        feats, y = self.preprocessor(wav, y)

        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)
        y_pred_logit = self.model(feats, labels, return_logits=True)

        loss = fn.binary_cross_entropy_with_logits(y_pred_logit, y)

        y_pred = self.postprocessor(y_pred_logit.sigmoid())
        self.update_sed_metrics(
            rearrange(y_pred, "b c t -> (b c) t").float(),
            rearrange(y, "b c t -> (b c) t").float(),
            filenames,
        )

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
            logx=feats["logx"][0].detach(),
            y_pred=y_pred[0].detach(),
            y=y[0].detach(),
            labels=labels[0],
        )
