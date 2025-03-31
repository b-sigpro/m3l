# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan

from pathlib import Path
import sys

import torch
from torch import nn


class BEATsModel(nn.Module):
    """
    Wrapper class for the BEATs audio representation model.

    This class loads a pretrained BEATs model for extracting features from raw audio waveforms.
    """

    def __init__(
        self,
        beats_path: str | Path = "./local/src/unilm/beats/",
        checkpoint_path: str | Path = "./local/src/unilm/beats/BEATs_iter3_plus_AS2M.pt",
    ):
        """
        Initializes the BEATs model with the specified configuration and checkpoint.

        Args:
            beats_path (str | Path): Path to the BEATs source directory.
            checkpoint_path (str | Path): Path to the pretrained BEATs model checkpoint.
        """

        super().__init__()

        sys.path.append(str(beats_path))
        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(checkpoint_path)

        cfg = BEATsConfig(checkpoint["cfg"])

        self.beats = BEATs(cfg)  # .cuda()
        self.beats.load_state_dict(checkpoint["model"])

        self.beats.eval()
        for param in self.beats.parameters():
            param.requires_grad = False

    @torch.no_grad
    def forward(self, feats: dict[str, torch.Tensor]):
        return self.beats.extract_features(feats["wav"])[0]
