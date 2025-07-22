import torch
import torch.nn as nn

import torch.nn.functional as F
import math

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "Default"
    vocab_size: int = 71
    block_size: int = 2048
    n_head: int = 4
    n_layers: int = 2
    dropout: int = 0.1
    k: int = 64  # k needs to be divisible by n_head


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
        h = self.heads
        k = self.k
        s = k // h

        b, N, _ = x.shape
        K = self.tokeys(x).view((b, N, h, s)).permute(0, 2, 1, 3)  # b, h, N, dk
        Q = self.toqueries(x).view((b, N, h, s)).permute(0, 2, 1, 3)  # b, h, N, dk

        raw_weights = Q @ K.transpose(-1, -2) / math.sqrt(s)  # b, h, N, N
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            raw_weights = raw_weights.masked_fill(mask == 0, float("-inf")).view(
                b * h, N, N
            )
        attention_weights = torch.softmax(raw_weights, dim=-1)

        V = self.tovalues(x).view(b, N, h, s).permute(0, 2, 1, 3)  # b, h, N, s
        heads_output = (attention_weights @ V).permute(0, 2, 1, 3)  # b, N, h, s
        heads_output.view(b, N, k)

        return self.unifyheads(heads_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, block_size):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            block_size - Maximum length of a sequence to expect.
        """
        super().__init__()

        pe = torch.zeros(block_size, d_model)
        position = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Modulo 3 and 6 encoding
        pe[:, : d_model // 4] += torch.sin(2 * math.pi * position / 3)
        pe[:, d_model // 4 : d_model // 2] += torch.cos(2 * math.pi * position / 6)

        # persistent=False tells PyTorch to not add the buffer to the state dict
        # (e.g. when we save the model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, k, heads, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(k, heads)

        self.dropout = torch.nn.Dropout(p=dropout)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.GELU(), nn.Linear(4 * k, k))

    def forward(self, x, mask=None):
        # pre-norm
        x = self.norm1(x)
        attended = self.attention(x, mask=mask)
        x = x + self.dropout(attended)
        x = self.norm2(x)
        fedforward = self.ff(x)
        result = x + self.dropout(fedforward)
        return result


class ChessNet(nn.Module):
    def __init__(self, config=ModelConfig):
        """
        Next-token logits
        """
        super().__init__()
        self.config = config
        self.embedder = nn.Embedding(
            config.vocab_size + 1, config.k, padding_idx=config.vocab_size
        )
        self.pe = PositionalEncoding(config.k, config.block_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.k, config.n_head, dropout=config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.l4 = nn.Linear(config.k, config.vocab_size)

    def forward(self, x, mask=None):
        # x:  b, T
        r = self.embedder(x)  # (b, T, k)
        r = self.pe(r)  # (b, T, k)
        for block in self.blocks:
            r = block(r, mask=mask)  #  (b, T, k)
        r = self.l4(r)  #  (b, T, k)
        return r
