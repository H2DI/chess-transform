import torch
from torch import nn

import torch.nn.functional as F
import math

from dataclasses import dataclass


# @dataclass
# class ModelConfig:
#     block_size = 256
#     vocab_size = 66
#     n_layer = 128
#     n_head = 6
#     n_embd = 284


@dataclass
class ModelConfig:
    vocab_size = 66
    block_size = 1024
    n_layer = 128
    n_head = 8
    k = 256  # k needs to be divisible by n_head


class SelfAttention(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.k = k
        self.heads = heads

        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)

        self.unifyheads = nn.Linear(k, k)

    def forward(self, x, mask=None):
        b, t, k = x.shape

        h = self.heads
        s = k // h

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        raw_weights = torch.bmm(keys, queries.transpose(1, 2))

        keys = keys.view(b, t, h, s).transpose(1, 2)
        queries = queries.view(b, t, h, s).transpose(1, 2)
        values = values.view(b, t, h, s).transpose(1, 2)

        keys = keys.contiguous().view(b * h, t, s)
        queries = queries.contiguous().view(b * h, t, s)
        values = values.contiguous().view(b * h, t, s)

        raw_weights = torch.bmm(queries, keys.transpose(-1, -2))

        if mask is not None:
            max_length = max(mask.shape)
            mask = mask.unsqueeze(1).unsqueeze(1).repeat(1, h, max_length, 1)
            raw_weights = raw_weights.view(b, h, t, t)
            raw_weights = raw_weights.masked_fill(mask == 0, float("-inf")).view(
                b * h, t, t
            )

        weights = F.softmax(raw_weights, dim=2)
        ys = torch.bmm(weights, values)

        ys = ys.view(b, h, t, s).transpose(1, 2)
        ys = ys.contiguous().view(b, t, k)

        return self.unifyheads(ys)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, block_size):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Modulo 2 and 4 encoding
        pe[:, : d_model // 4] += torch.sin(2 * math.pi * position / 2)
        pe[:, d_model // 4 : d_model // 2] += torch.cos(2 * math.pi * position / 4)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads)

        self.dropout = torch.nn.Dropout(p=0.1)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.GELU(), nn.Linear(4 * k, k))

    def forward(self, x: torch.Tensor, mask=None):
        attended = self.attention(x, mask=mask)
        attended = self.dropout(attended)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        result = self.norm2(fedforward + x)
        return result


class ChessNet(nn.Module):
    def __init__(self, config=ModelConfig):
        super().__init__()
        self.config = config
        self.embedder = nn.Embedding(
            config.vocab_size + 1, config.k, padding_idx=config.vocab_size
        )
        self.pe = PositionalEncoding(config.k, config.block_size)
        self.l1 = TransformerBlock(config.k, config.n_head)
        self.l2 = TransformerBlock(config.k, config.n_head)
        self.l3 = TransformerBlock(config.k, config.n_head)
        self.l4 = nn.Linear(config.k, config.vocab_size)

    def forward(self, x, mask=None):
        r = self.embedder(x)  # (batch_size, seq_len, k)
        r = self.pe(r)  # (batch_size, seq_len, k)
        r = self.l1(r, mask=mask)  # (batch_size, seq_len, k)
        r = self.l2(r, mask=mask)  # (batch_size, seq_len, k)
        r = self.l3(r, mask=mask)  # (batch_size, seq_len, k)
        r = self.l4(r)  #  (batch_size, seq_len, vocab_size)
        # r = torch.sum(r, axis=1)
        return r


class Benchmark(nn.module):
    def __init__(self, move=None):
        super().__init__()
        if move is None:
            move = torch.tensor([35])  # e2
        self.move = move

    def forward(self, _):
        return self.move
