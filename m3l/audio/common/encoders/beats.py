# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import remove_weight_norm as _remove_weight_norm


class BEATsModel(nn.Module):
    """
    Wrapper class for the BEATs audio representation model.

    This class loads a pretrained BEATs model for extracting features from raw audio waveforms.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "./3rd/unilm/beats/BEATs_iter3_plus_AS2M.pt",
        freeze: bool = True,
        overwrite_encoder_layerdrop: float | None = None,
        remove_weight_norm: bool = False,
    ):
        """
        Initializes the BEATs model with the specified configuration and checkpoint.

        Args:
            beats_path (str | Path): Path to the BEATs source directory.
            checkpoint_path (str | Path): Path to the pretrained BEATs model checkpoint.
            freeze (bool): Whether to freeze the pretrained model weights.
            overwrite_encoder_layerdrop (float | None): Override value for the encoder layerdrop ratio.
            remove_weight_norm (bool): Whether to remove weight normalization from the model.
        """

        super().__init__()
        self.freeze = freeze

        from m3l.third.beats.BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(checkpoint_path)

        cfg = BEATsConfig(checkpoint["cfg"])

        if overwrite_encoder_layerdrop is not None:
            cfg.encoder_layerdrop = overwrite_encoder_layerdrop  # override encoder layerdrop ratio

        self.beats = BEATs(cfg)  # .cuda()
        self.beats.load_state_dict(checkpoint["model"])

        if remove_weight_norm:
            _remove_weight_norm(self.beats.encoder.pos_conv[0], name="weight")

        if freeze:
            for param in self.beats.parameters():
                param.requires_grad = False

    def forward(self, feats: dict[str, torch.Tensor]):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            return self.beats.extract_features(feats["wav"])[0]
