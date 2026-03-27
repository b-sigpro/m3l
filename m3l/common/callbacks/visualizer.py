# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from matplotlib import pyplot as plt
import numpy as np

import lightning as lt


class VisualizerCallback(lt.Callback):
    """Callback to visualize model predictions during training and validation.

    Args:
        label_names (list[str]): List of class labels.
        time_stretch (int, optional): Factor to stretch time axis of predictions. Defaults to 1.

    Methods:
        on_validation_start: Called when the validation loop starts. Plots spectrogram, ground truth,
            and predictions for a random batch, and logs the figure to the experiment logger.
        on_validation_end: Called when the validation loop ends. Runs the same visualization
            as ``on_validation_start`` but with the "validation" tag.
    """

    def __init__(self, label_names: list[str], time_stretch: int = 1):
        self.label_names = np.asarray(label_names)
        self.time_stretch = time_stretch

    def on_validation_start(self, trainer: lt.Trainer, pl_module: Any, tag: str = "training"):
        if not hasattr(pl_module, "dump"):
            return

        dump = pl_module.dump

        # numpyize dumped variables
        B, *_ = dump.logx.shape
        b = np.random.choice(B)

        dump = dump.__class__(**{k: v[b].cpu().numpy() for k, v in vars(dump).items()})

        # plot predictions
        fig, axs = plt.subplots(3, 1, figsize=[8, 8], sharex=True)

        axs[0].imshow(dump.logx, origin="lower", aspect="auto")

        lorder = np.argsort(10 * dump.y.sum(axis=-1) + dump.y_pred.sum(axis=-1))[-16:]
        kwargs = dict(aspect="auto", interpolation="none", origin="lower", vmin=0, vmax=1)
        axs[1].imshow(dump.y[lorder].repeat(self.time_stretch, axis=-1), **kwargs)
        axs[2].imshow(dump.y_pred[lorder].repeat(self.time_stretch, axis=-1), **kwargs)

        for ax in axs[1:]:
            ax.set_yticks(range(len(lorder)))
            ax.set_yticklabels([self.label_names[ll] for ll in lorder], ha="right", fontsize=10)

        fig.tight_layout(pad=0.1)
        pl_module.logger.experiment.add_figure(f"{tag}/xt", fig, global_step=trainer.current_epoch)
        plt.close(fig)

    def on_validation_end(self, trainer: lt.Trainer, pl_module: Any):
        self.on_validation_start(trainer, pl_module, tag="validation")
