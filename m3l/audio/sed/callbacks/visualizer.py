from typing import Any

from matplotlib import pyplot as plt
import numpy as np

import lightning as lt


class VisualizerCallback(lt.Callback):
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
