# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from matplotlib import pyplot as plt
import numpy as np

import torch

import lightning as lt


class VisualizerCallback(lt.Callback):
    """Callback to visualize model predictions during validation.

    This callback generates spectrogram and bar chart plots of the model's
    predictions and logs them to the experiment logger (e.g., TensorBoard).
    It compares ground truth labels with predicted scores and displays
    the top-ranked events.

    Args:
        label_names (list[str]): List of class label names.
    """

    def __init__(self, label_names: list[str]):
        self.label_names = np.asarray(label_names)

    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        dump = pl_module.dump

        # numpyize dumped variables
        B, *_ = dump.logx.shape

        dump.y_pred = torch.sigmoid(dump.y_pred)
        dump = dump.__class__(**{k: v.cpu().to(torch.float32).numpy() for k, v in vars(dump).items()})

        # plot predictions
        fig, axs = plt.subplots(3, 1, figsize=[8, 8])

        axs[0].imshow(dump.logx, origin="lower", aspect="auto")

        lorder = np.argsort(10 * dump.y + dump.y_pred)[-20:][::-1]

        axs[1].bar(range(len(lorder)), dump.y[lorder], align="center")
        axs[2].bar(range(len(lorder)), dump.y_pred[lorder], align="center")

        for ax in axs[1:]:
            ax.set_ylim(0, 1.1)

        axs[-1].set_xticks(range(len(lorder)))
        axs[-1].set_xticklabels([self.label_names[ll] for ll in lorder], rotation=45, ha="right", fontsize=10)

        axs[1].sharex(axs[2])
        axs[1].label_outer()

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/xt", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
