#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import logging

class DecoderLayer(nn.Module):
    """Single decoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)


    """

    def __init__(
        self,
        size,
        self_attn_cn,
        self_attn_en,
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn_cn = self_attn_cn
        self.self_attn_en = self_attn_en
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1_cn = LayerNorm(size)
        self.norm1_en = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1_cn = nn.Linear(size + size, size)
            self.concat_linear1_en = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, token_id, chn_start, chn_end, eng_start ,eng_end, sym_low, sym_up, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        # 0/1 tensor
        cn_location = (token_id >= chn_start).type(torch.float32) * \
            (token_id <= chn_end).type(torch.float32)
        en_location = (token_id >= eng_start).type(torch.float32) * \
            (token_id <= eng_end).type(torch.float32)
        sym_location = (token_id <= sym_low).type(torch.float32) * \
            (token_id >= sym_up).type(torch.float32)
        

        _, _, hidden_dim = tgt.size()
        cn_mask = (cn_location + en_location * 0.1 + sym_location * 0.5).unsqueeze(2).repeat(1, 1, hidden_dim)
        en_mask = (en_location + cn_location * 0.1 + sym_location * 0.5).unsqueeze(2).repeat(1, 1, hidden_dim)
        #LA_tgt (Language-Aware)
        cn_tgt = cn_mask * tgt
        en_tgt = en_mask * tgt

        cn_residual = cn_tgt
        en_residual = en_tgt

        if self.normalize_before:
            cn_tgt = self.norm1_cn(cn_tgt)
            en_tgt = self.norm1_en(en_tgt)

        if cache is None:
            cn_tgt_q = cn_tgt
            cn_tgt_q_mask = tgt_mask
            en_tgt_q = en_tgt
            en_tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            cn_tgt_q = cn_tgt[:, -1:, :]
            en_tgt_q = en_tgt[:, -1:, :]
            cn_residual = cn_residual[:, -1:, :]
            en_residual = en_residual[:, -1:, :]
            cn_tgt_q_mask = None
            en_tgt_q_mask = None
            if tgt_mask is not None:
                cn_tgt_q_mask = tgt_mask[:, -1:, :]
                en_tgt_q_mask = tgt_mask[:, -1:, :]

        #cn-self
        if self.concat_after:
            cn_tgt_concat = torch.cat(
                (cn_tgt_q, self.self_attn_cn(cn_tgt_q, cn_tgt, cn_tgt, None)), dim=-1
            )
            cn_x = cn_residual + self.concat_linear1_cn(cn_tgt_concat)
        else:
            cn_x = cn_residual + self.dropout(self.self_attn_cn(cn_tgt_q, cn_tgt, cn_tgt, None))
        #en-self
        if self.concat_after:
            en_tgt_concat = torch.cat(
                (en_tgt_q, self.self_attn_en(en_tgt_q, en_tgt, en_tgt, None)), dim=-1
            )
            en_x = en_residual + self.concat_linear1_en(en_tgt_concat)
        else:
            en_x = en_residual + self.dropout(self.self_attn_en(en_tgt_q, en_tgt, en_tgt, None))

        if not self.normalize_before:
            cn_x = self.norm1_cn(cn_x)
            en_x = self.norm1_en(en_x)
        #concat
        x = cn_x + en_x

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
            )
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask, token_id, chn_start, chn_end, eng_start ,eng_end, sym_low, sym_up
