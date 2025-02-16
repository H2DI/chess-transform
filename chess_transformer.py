import torch
from torch import nn

import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, k=64, heads=8):
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
    def __init__(self, d_model, max_len=600):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, k=64, heads=8):
        super().__init__()

        self.attention = SelfAttention(k=k, heads=heads)

        self.dropout = torch.nn.Dropout(p=0.1)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, k))

    def forward(self, x: torch.Tensor, mask=None):
        attended = self.attention(x, mask=mask)
        attended = self.dropout(attended)
        x = self.norm1(attended + x)
        fedforward = self.ff(x)
        result = self.norm2(fedforward + x)
        return result


class ChessNet(nn.Module):
    def __init__(self, n_vocab, k=64, heads=8):
        super().__init__()
        self.k = k
        self.heads = heads

        self.embedder = nn.Embedding(n_vocab + 1, k, padding_idx=n_vocab)
        self.pe = PositionalEncoding(k)
        self.l1 = TransformerBlock(k, heads)
        self.l2 = TransformerBlock(k, heads)
        self.l3 = TransformerBlock(k, heads)
        self.l4 = nn.Linear(k, n_vocab)

    def forward(self, x, mask=None):
        r = self.embedder(x)
        r = self.pe(r)
        r = self.l1(r, mask=mask)
        r = self.l2(r, mask=mask)
        r = self.l3(r, mask=mask)
        r = self.l4(r)
        r = torch.sum(r, axis=1)
        return r


class ChessNet2(nn.Module):
    def __init__(self, n_vocab, k=64):
        super().__init__()
        self.n_vocab = n_vocab
        self.k = k

        self.embedder = nn.Embedding(n_vocab + 1, k, padding_idx=n_vocab)
        nn.Transformer()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=k, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=6
        )
        self.prob_layer = nn.Linear(k, n_vocab)

    def forward(self, x, mask=None):
        x = self.embedder(x)
        x = self.transformer_encoder(x, tgt_mask=mask)
        x = self.prob_layer(x)
        x = torch.sum(x, axis=1)
        return x
