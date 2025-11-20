import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_seq.models import GQARope, RoPE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def forward(self, x):
        # x: (..., 2 * d)
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x2) * x1


class TinyAttentionBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4, heads: int = 2):
        super().__init__()
        inner_dim = dim * hidden_mult
        self.norm1 = RMSNorm(dim)
        self.rope = RoPE(dim, 200)
        self.fc1 = GQARope(dim, heads=heads, groups=1)
        self.norm2 = RMSNorm(dim)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 2 * inner_dim, bias=False),
            SwiGLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        # x: (B, T,  D)
        h = self.norm1(x + self.fc1(x, self.rope))
        return self.norm2(h + self.fc2(h))


class TinyMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        inner_dim = dim * hidden_mult
        self.norm1 = RMSNorm(dim)
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 2 * inner_dim, bias=False),
            SwiGLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )
        self.norm2 = RMSNorm(dim)
        self.fc2 = nn.Sequential(
            nn.Linear(dim, 2 * inner_dim, bias=False),
            SwiGLU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        # x: B, D
        h = self.norm1(x + self.fc1(x))
        return self.norm2(h + self.fc2(h))


class ReasoningNet(nn.Module):
    """
    A tiny 2-layer network used inside TRM.
    """

    def __init__(self, dim: int, depth: int = 2, hidden_mult: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [TinyMLPBlock(dim, hidden_mult=hidden_mult) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
