import torch
from torch import nn
from torch.nn import functional as fn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from aiaccel.torch.lightning import OptimizerConfig

from m3l.audio.common.preprocessors.preprocessor import Preprocessor
from m3l.audio.sed.tasks.sed_task import SEDTask, Snapshot


class WeakEMASEDTask(SEDTask):
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
        beta: float = 1.0,
        calc_event_metrics: bool = True,
    ):
        super().__init__(
            preprocessor,
            model,
            postprocessor,
            n_class,
            label_names,
            time_resolution,
            val_duration,
            val_groundtruth,
            optimizer_config,
            freezed_model,
            threshold,
            alpha,
            calc_event_metrics,
        )

        self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)
        self.ema_model.requires_grad_(False)

        self.beta = beta

    def _take_str(self, x: torch.Tensor, y_str: torch.Tensor) -> torch.Tensor:
        n_str = len(y_str)

        return x[:n_str]

    def _take_wek(self, x: torch.Tensor, y_str: torch.Tensor, y_wek: torch.Tensor) -> torch.Tensor:
        n_str = len(y_str)
        n_wek = len(y_wek)

        return x[n_str : n_str + n_wek]

    def on_train_batch_start(
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]],
        batch_idx: int,
    ):
        self.ema_model.update_parameters(self.model)

    def training_step(  # type: ignore
        self,
        batch: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]],
        batch_idx: int,
    ):
        self.dump = None

        wav_str, y_str, _ = batch["strong"]
        wav_wek, y_wek, _ = batch["weak"]
        wav_uns, *_ = batch["unsup"]

        # preprocess data
        feats_str, y_str = self.preprocessor(wav_str, y_str)
        feats_wek, y_wek = self.preprocessor(wav_wek, y_wek)
        feats_uns, _ = self.preprocessor(wav_uns)

        feats = {k: torch.concat((v_str, feats_wek[k], feats_uns[k])) for k, v_str in feats_str.items()}

        # predict labels
        if self.freezed_model is not None:
            feats["emb"] = self.freezed_model(feats)
        y_pred_str, y_pred_wek = self.model(feats, output_weak=True)

        with torch.no_grad():
            y_ema_str, y_ema_wek = self.ema_model({k: v.detach() for k, v in feats.items()}, output_weak=True)

        # calculate loss
        loss_str = fn.binary_cross_entropy(self._take_str(y_pred_str, y_str), y_str)
        loss_wek = fn.binary_cross_entropy(self._take_wek(y_pred_wek, y_str, y_wek), y_wek)

        loss_ora = loss_str + self.alpha * loss_wek

        loss_ema_str = fn.mse_loss(y_pred_str, y_ema_str)
        loss_ema_wek = fn.mse_loss(y_pred_wek, y_ema_wek)

        loss_ema = loss_ema_str + self.alpha * loss_ema_wek

        loss = loss_ora + self.beta * loss_ema

        # logging
        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                "training/loss_str": loss_str,
                "training/loss_wek": loss_wek,
                "training/loss_ema_str": loss_ema_str,
                "training/loss_ema_wek": loss_ema_wek,
                "training/loss": loss,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        self.dump = Snapshot(
            logx=self._take_str(feats["logx"], y_str).detach(),
            y_pred=self._take_str(y_pred_str, y_str).detach(),
            y=y_str.detach(),
        )

        return loss
