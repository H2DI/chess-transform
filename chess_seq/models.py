import torch
import torch.nn as nn

import math

from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = "Default"
    vocab_size: int = 71
    block_size: int = 512
    n_head: int = 4
    n_layers: int = 4
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

    def forward(self, x, rope, mask=None):
        h = self.heads
        k = self.k
        s = k // h

        b, N, _ = x.shape

        K = rope(self.tokeys(x).view((b, N, h, s)).permute(0, 2, 1, 3))  # b, h, N, dk
        Q = rope(self.toqueries(x).view((b, N, h, s)).permute(0, 2, 1, 3))

        raw_weights = Q @ K.transpose(-1, -2) / math.sqrt(s)  # b, h, N, N
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
            raw_weights = raw_weights.masked_fill(mask == 0, float("-inf")).view(
                b, h, N, N
            )
        attention_weights = torch.softmax(raw_weights, dim=-1)

        V = self.tovalues(x).view(b, N, h, s).permute(0, 2, 1, 3)  # b, h, N, s
        heads_output = (attention_weights @ V).permute(0, 2, 1, 3)  # b, N, h, s

        return self.unifyheads(heads_output.reshape(b, N, k))


class RoPE(nn.Module):
    def __init__(self, d_model, block_size):
        super().__init__()
        assert d_model % 2 == 0

        self.d_model = d_model
        self.block_size = block_size

        half_dim = d_model // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        ).unsqueeze(0)  # 1, k // 2
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        angles = self._get_angles(block_size)  # block_size, k //2

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x):
        b, h, N, k = x.shape
        x_reshaped = x.view(b, h, N, k // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        if N <= self.block_size:
            cos = self.cos[:N].unsqueeze(0).unsqueeze(0)
            sin = self.sin[:N].unsqueeze(0).unsqueeze(0)
        else:
            cos, sin = self._recompute_cossin(N, x.device)

        rotated = torch.stack([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
        return rotated.reshape(b, h, N, k)

    def _get_angles(self, N, device=None):
        positions = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(
            1
        )  # N_max, 1
        return positions @ self.inv_freq  # N_max, k //2

    def _recompute_cossin(self, N, device=None):
        angles = self._get_angles(N, device)
        return torch.cos(angles), torch.sin(angles)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(k, heads)

        self.dropout = torch.nn.Dropout(p=dropout)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.GELU(), nn.Linear(4 * k, k))

    def forward(self, x, rope, mask=None):
        # pre-norm
        x = self.norm1(x)
        attended = self.attention(x, rope, mask=mask)
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
        self.rope = RoPE(config.k // config.n_head, config.block_size)
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
        for block in self.blocks:
            r = block(r, self.rope, mask=mask)  #  (b, T, k)
        r = self.l4(r)  #  (b, T, k)
        return r
