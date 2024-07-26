import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from torchtext import data, datasets, vocab

# from torchtext.legacy import data, datasets, vocab

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

# b, t, k = 1, 5, 8
# x = torch.zeros(b, t, k)


# key_weights = torch.ones(b, k, k)
# query_weights = torch.ones(b, k, k)
# value_weights = torch.ones(b, k, k)

# key = torch.bmm(x, key_weights)
# query = torch.bmm(x, query_weights)
# value = torch.bmm(x, value_weights)

# raw_weights = torch.bmm(key, query.transpose(1, 2))
# weights = F.softmax(raw_weights, dim=2)
# print(weights.shape)
# y = torch.bmm(weights, value)
# print(y.shape)


class SelfAttention(nn.Module):

    def __init__(self, k, heads=4, mask=False):
        super().__init__()
        self.k = k
        self.heads = heads

        self.tokeys = nn.Linear(k, k, bias=False)
        self.toqueries = nn.Linear(k, k, bias=False)
        self.tovalues = nn.Linear(k, k, bias=False)

        self.unifyheads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.shape

        h = self.heads
        s = h // k

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

        raw_weights = torch.bmm(queries, keys.transpose(2, 3))

        if self.mask and mask is not None:
            mask = mask.unsqueeze(1).repeat(1, h, 1, 1)  # Extend mask for all heads
            raw_weights = raw_weights.view(b, h, t, t)
            raw_weights = raw_weights.masked_fill(mask == 0, float("-inf")).view(
                b * h, t, t
            )

        weights = F.softmax(raw_weights, dim=2)
        ys = torch.bmm(weights, values)

        ys = ys.view(b, h, t, s).transpose(1, 2)
        ys = ys.contiguous().view(b, t, k)

        return self.unifyheads(ys)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


# https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/
# transformer_positional_encoding_graph.ipynb


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return pos_encoding
