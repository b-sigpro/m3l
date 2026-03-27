# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# Copyright (c) 2025, National Institute of Advanced Industrial Science and Technology (AIST)
# Copyright (c) 2001-2025, PyTorch Contributors
#
# SPDX-License-Identifier: MIT AND BSD-3-Clause


from functools import partial

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Transformer encoder layer that supports positional encoding in query and key.
    This layer allows for the addition of positional encodings to the input tensor
    before the self-attention mechanism, which is useful for tasks where the order
    of the input sequence matters. Basic usage is the same as the standard
    `torch.nn.TransformerEncoderLayer`, but with the addition of positional encodings.

    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the MultiHeadAttention models (required).
        dim_feedforward (int, optional): The dimension of the feedforward network model (default= 2048).
        dropout (float, optional): The dropout value (default = 0.1).
        activation (str, optional): The activation function of intermediate layer, relu or gelu (default = relu).
        layer_norm_eps (float, optional): The epsilon used in layer normalization (default = 1e-5).
        batch_first (bool, optional): If True, then the input and output tensors are
            provided as (batch, seq, feature) (default = False).
        norm_first (bool, optional): If True, performs layer normalization before the
            self-attention and feedforward blocks (default = False).
        is_causal (bool, optional): If True, applies causal attention mask (default = False).
    Inputs:
        - src: the sequence to the encoder layer (required).
        - pos: the positional encoding to be added to the input (optional).
        - src_mask: the mask for the src sequence (optional).
        - src_key_padding_mask: the mask for the src keys per batch (optional).
        - is_causal: if True, applies causal attention mask (default = False).
    Outputs:
        - the output tensor of the encoder layer with the same shape as the input.

    Example:
        >>> import torch
        >>> from aiaccel.torch.nn.transformer import TransformerEncoderLayer
        >>> layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.randn(10, 32, 512)  # (sequence_length, batch_size, feature_size)
        >>> pos = torch.randn(10, 32, 512)  # (sequence_length, batch_size, feature_size)
        >>> output = layer(src, pos=pos)
        >>> print(output.shape)  # Should be (10, 32, 512)
    """

    def forward(  # type: ignore[override]
        self,
        src: torch.Tensor,
        pos: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        _sa_block = partial(
            self._sa_block,
            pos=pos,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )

        return self._forward_impl(
            src,
            _sa_block=_sa_block,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        _sa_block: callable,
    ):
        if self.norm_first:
            x = x + _sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + _sa_block(x))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(  # type: ignore[override]
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        qk = (x + pos) if pos is not None else x
        x = self.self_attn(
            query=qk,
            key=qk,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        return self.dropout1(x)  # type: ignore[no-any-return]


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Transformer decoder layer that supports positional encoding in query and key.
    This layer allows for the addition of positional encodings to the input tensor
    before the self-attention and multi-head attention mechanisms, which is useful
    for tasks where the order of the input sequence matters. Basic usage is the
    same as the standard `torch.nn.TransformerDecoderLayer`, but with the addition
    of positional encodings.

    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the MultiHeadAttention models (required).
        dim_feedforward (int, optional): The dimension of the feedforward network model (default= 2048).
        dropout (float, optional): The dropout value (default = 0.1).
        activation (str, optional): The activation function of intermediate layer, relu or gelu (default = relu).
        layer_norm_eps (float, optional): The epsilon used in layer normalization (default = 1e-5).
        batch_first (bool, optional): If True, then the input and output tensors are
            provided as (batch, seq, feature) (default = False).
        norm_first (bool, optional): If True, performs layer normalization before the
            self-attention and feedforward blocks (default = False).
        is_causal (bool, optional): If True, applies causal attention mask (default = False).
    Inputs:
        - tgt: the sequence to the decoder layer (required).
        - memory: the sequence from the last layer of the encoder (required).
        - tgt_pos: the positional encoding to be added to the target input (optional).
        - memory_pos: the positional encoding to be added to the memory input (optional).
        - tgt_mask: the mask for the target sequence (optional).
        - memory_mask: the mask for the memory sequence (optional).
        - tgt_key_padding_mask: the mask for the target keys per batch (optional).
        - memory_key_padding_mask: the mask for the memory keys per batch (optional).
        - tgt_is_causal: if True, applies causal attention mask to the target sequence (default = False).
        - memory_is_causal: if True, applies causal attention mask to the memory sequence (default = False).
    Outputs:
        - the output tensor of the decoder layer with the same shape as the target input.

    Example:
        >>> import torch
        >>> from aiaccel.torch.nn.transformer import TransformerDecoderLayer
        >>> layer = TransformerDecoderLayer(d_model=512, nhead=8)
        >>> tgt = torch.randn(10, 32, 512)  # (sequence_length, batch_size, feature_size)
        >>> memory = torch.randn(10, 32, 512)  # (sequence_length, batch_size, feature_size)
        >>> tgt_pos = torch.randn(10, 32, 512)  # (sequence_length, batch_size, feature_size)
        >>> memory_pos = torch.randn(10, 32, 512)  # (sequence_length, batch_size, feature_size)
        >>> output = layer(tgt, memory, tgt_pos=tgt_pos, memory_pos=memory_pos)
        >>> print(output.shape)  # Should be (10, 32, 512)
    """

    def forward(  # type: ignore[override]
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        _sa_block = partial(
            self._sa_block,
            pos=tgt_pos,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=tgt_is_causal,
        )

        _mha_block = partial(
            self._mha_block,
            mem=memory,
            pos=tgt_pos,
            mem_pos=memory_pos,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            is_causal=memory_is_causal,
        )

        return self._forward_impl(
            tgt,
            _sa_block=_sa_block,
            _mha_block=_mha_block,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        _sa_block: callable,
        _mha_block: callable,
    ):
        if self.norm_first:
            x = x + _sa_block(self.norm1(x))
            x = x + _mha_block(self.norm2(x))
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + _sa_block(x))
            x = self.norm2(x + _mha_block(x))
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(  # type: ignore[override]
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        qk = (x + pos) if pos is not None else x
        x = self.self_attn(
            query=qk,
            key=qk,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]

        return self.dropout1(x)  # type: ignore[no-any-return]

    def _mha_block(  # type: ignore[override]
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        pos: torch.Tensor | None,
        mem_pos: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            query=(x + pos) if pos is not None else x,
            key=(mem + mem_pos) if mem_pos is not None else mem,
            value=mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]

        return self.dropout2(x)  # type: ignore[no-any-return]
