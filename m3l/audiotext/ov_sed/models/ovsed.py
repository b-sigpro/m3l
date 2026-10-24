# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# MIT License
# Copyright (c) 2025 National Institute of Advanced Industrial Science and Technology (AIST), Japan


from einops import rearrange

import torch
from torch import nn

from einops.layers.torch import Rearrange

from aiaccel.torch.nn import SinusoidalPositionalEmbedding

from m3l.common.nn.transformer import TransformerEncoderLayer


class CrossAttnBlock(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, kernel_size: int, dropout: float = 0.2, tau: float = 20
    ):
        super().__init__()

        self.pe = SinusoidalPositionalEmbedding(d_model, tau=tau, normalize=True)

        self.cross_attn_a = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross_attn_l = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.self_attn_a = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.self_attn_l = TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
        )

        self.ffn_a = nn.Sequential(
            nn.LayerNorm(d_model),
            Rearrange("b t c -> b c t"),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, dim_feedforward, kernel_size, padding=(kernel_size - 1) // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_feedforward, d_model, kernel_size, padding=kernel_size // 2),
            Rearrange("b c t -> b t c"),
        )

        self.norm1_a = nn.LayerNorm(d_model)

    def forward(self, emb_audio: torch.Tensor, emb_label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        emb_audio: (B, T, C)
        emb_label: (B, L, C)
        """

        emb_audio_pe = emb_audio + self.pe(emb_audio)
        emb_audio_, _ = self.cross_attn_a(emb_audio_pe, emb_label, emb_label, need_weights=False)
        emb_label_, _ = self.cross_attn_l(emb_label, emb_audio_pe, emb_audio, need_weights=False)

        emb_audio = emb_audio + emb_audio_
        emb_label = emb_label + emb_label_

        emb_audio = self.self_attn_a(emb_audio, self.pe(emb_audio))
        emb_label = self.self_attn_l(emb_label)

        emb_audio = self.norm1_a(emb_audio + self.ffn_a(emb_audio))

        return emb_audio, emb_label


class OpenVocabularySEDModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        kernel_size: int,
        n_blocks: int,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder

        self.norm1_a = nn.LayerNorm(d_model)
        self.norm1_l = nn.LayerNorm(d_model)

        self.attn_blocks = nn.ModuleList(
            [CrossAttnBlock(d_model, nhead, dim_feedforward, kernel_size) for _ in range(n_blocks)]
        )

        self.clip_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        labels: list[list[str]],
        return_all: bool = False,
        return_logits: bool = False,
        return_weak: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb_audio = rearrange(self.audio_encoder(feats), "b c t -> b t c")
        emb_label = self.text_encoder(labels)

        emb_audio = self.norm1_a(emb_audio)
        emb_label = self.norm1_l(emb_label)

        emb_audio_list = []
        emb_label_list = []
        for block in self.attn_blocks:
            emb_audio, emb_label = block(emb_audio, emb_label)
            emb_audio_list.append(emb_audio)
            emb_label_list.append(emb_label)

        if return_all:
            emb_audio = torch.stack(emb_audio_list, dim=0)
            emb_label = torch.stack(emb_label_list, dim=0)
            y_str = torch.einsum("nbtc,nblc->nblt", emb_audio, emb_label)
        else:
            y_str = torch.einsum("btc,blc->blt", emb_audio, emb_label)

        y_weak = self.clip_head(emb_label)

        y_weak = y_weak if return_logits else y_weak.sigmoid()
        y_str = y_str if return_logits else y_str.sigmoid()

        if return_weak:
            return y_str, y_weak.squeeze(-1)
        else:
            return y_str
