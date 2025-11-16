import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    name: str = "Default"
    vocab_size: int = 71  # 151_936
    block_size: int = 512
    n_head: int = 16
    n_layers: int = 28
    dropout: int = 0.0
    kv_groups: int = 2
    k: int = 1024  # k needs to be divisible by n_head
    special_freqs: List[float] = None

    encoder_path: str = "data/move_encoder.pkl"


# class SelfAttention(nn.Module):
#     def __init__(self, k, heads):
#         super().__init__()
#         self.k = k
#         self.heads = heads

#         self.tokeys = nn.Linear(k, k, bias=False)
#         self.toqueries = nn.Linear(k, k, bias=False)
#         self.tovalues = nn.Linear(k, k, bias=False)

#         self.unifyheads = nn.Linear(k, k)

#     def forward(self, x, rope, mask=None):
#         h = self.heads
#         k = self.k
#         dh = k // h

#         b, N, _ = x.shape

#         K = rope(self.tokeys(x).view((b, N, h, dh)).permute(0, 2, 1, 3))
#         Q = rope(self.toqueries(x).view((b, N, h, dh)).permute(0, 2, 1, 3))

#         raw_weights = Q @ K.transpose(-1, -2) / math.sqrt(dh)
#         if mask is not None:
#             mask = mask.unsqueeze(0).unsqueeze(0)
#             raw_weights = raw_weights.masked_fill(mask == 0, float("-inf")).view(
#                 b, h, N, N
#             )
#         attention_weights = torch.softmax(raw_weights, dim=-1)

#         V = self.tovalues(x).view(b, N, h, dh).permute(0, 2, 1, 3)
#         heads_output = (attention_weights @ V).permute(0, 2, 1, 3)

#         return self.unifyheads(heads_output.reshape(b, N, k))


# class AttentionRope(nn.Module):
#     def __init__(self, k, heads, dropout=0.1):
#         super().__init__()
#         self.heads = heads
#         self.head_dim = k // heads
#         self.qkv = nn.Linear(k, 3 * k, bias=False)
#         self.out = nn.Linear(k, k)

#         self.dropout_p = dropout

#     def forward(self, x, rope, is_causal=True):
#         b, T, k = x.shape
#         qkv = self.qkv(x).reshape(b, T, self.heads, 3, self.head_dim)
#         Q, K, V = qkv.unbind(dim=3)  # b, T, h, dh

#         Q = Q.transpose(1, 2)  # b, h, t, dh
#         K = K.transpose(1, 2)
#         V = V.transpose(1, 2)

#         Q, K = rope(Q), rope(K)
#         out = F.scaled_dot_product_attention(
#             Q,
#             K,
#             V,
#             attn_mask=None,
#             dropout_p=self.dropout_p if self.training else 0.0,
#             is_causal=is_causal,
#         )  # b, h, t, dh

#         out = out.transpose(1, 2).reshape(b, T, k)
#         return self.out(out)

# class TransformerBlock(nn.Module):
#     def __init__(self, k, heads, dropout=0.1):
#         super().__init__()

#         self.attention = AttentionRope(k, heads, dropout=dropout)

#         self.norm1 = nn.LayerNorm(k)
#         self.norm2 = nn.LayerNorm(k)

#         self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.GELU(), nn.Linear(4 * k, k))

#     def forward(self, x, rope, is_causal=True):
#         # pre-norm
#         x = self.norm1(x)
#         x = x + self.attention(x, rope, is_causal=is_causal)  # dropout included
#         x = self.norm2(x)
#         return x + self.ff(x)


class RoPE(nn.Module):
    def __init__(self, d_model, block_size, special_freqs=None):
        super().__init__()
        assert d_model % 2 == 0
        if special_freqs is None:
            special_freqs = [2 * math.pi / 3, 2 * math.pi / 6]

        self.d_model = d_model
        self.block_size = block_size

        half_dim = d_model // 2
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        ).unsqueeze(0)  # 1, k // 2

        for i, freq in enumerate(special_freqs):
            inv_freq[0, -i - 1] = freq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        angles = self._get_angles(block_size)  # block_size, k //2

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x):
        b, h, T, k = x.shape
        assert k % 2 == 0, f"Expected even embedding dim, got {k}"
        x_reshaped = x.view(b, h, T, k // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        if T <= self.block_size:
            cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
            sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        else:
            cos, sin = self._recompute_cossin(T, x.device)

        rotated = torch.stack([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
        return rotated.reshape(b, h, T, k)

    def _get_angles(self, N, device=None):
        positions = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
        # N, 1
        return positions @ self.inv_freq  # N, k //2

    def _recompute_cossin(self, N, device=None):
        angles = self._get_angles(N, device)
        return torch.cos(angles), torch.sin(angles)


class GQARope(nn.Module):
    def __init__(self, k, heads=16, groups=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.groups = groups
        self.head_dim = k // heads

        self.qkv = nn.Linear(k, k + 2 * k // groups, bias=False)

        self.unify_heads = nn.Linear(k, k)

        self.dropout_p = dropout

    def forward(self, x, rope, is_causal=True):
        b, T, k = x.shape
        h, groups = self.heads, self.groups
        hg = h // groups
        kh = k // h

        Q, K, V = torch.split(self.qkv(x), [k, k // groups, k // groups], dim=-1)

        Q = Q.view(b, T, h, kh).transpose(1, 2)
        K = K.reshape(b, T, hg, kh).transpose(1, 2)
        V = V.reshape(b, T, hg, kh).transpose(1, 2)

        K = K.unsqueeze(2).expand(b, hg, groups, T, kh).reshape(b, h, T, kh)
        V = V.unsqueeze(2).expand(b, hg, groups, T, kh).reshape(b, h, T, kh)

        Q, K = rope(Q), rope(K)

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # b, h, t, dh

        out = out.transpose(1, 2).reshape(b, T, k)
        return self.unify_heads(out)


class SwiGLU(nn.Module):
    # Replace MLP in Transformer block

    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)  # value path
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)  # gate path
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)  # projection back to model dim

    def forward(self, x):
        return self.w3(self.w1(x) * F.silu(self.w2(x)))


class ParallelBlock(nn.Module):
    def __init__(self, dim, n_heads, groups, dropout=0.0):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.attn = GQARope(dim, n_heads, groups, dropout=dropout)
        self.ff = SwiGLU(dim, hidden_dim=4 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope, is_causal=True):
        # Parallel residuals, no dropout
        h = self.norm(x)
        attn_out = self.attn(h, rope, is_causal=is_causal)
        ff_out = self.ff(h)
        return x + self.dropout(attn_out + ff_out)


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
        self.rope = RoPE(
            config.k // config.n_head,
            config.block_size,
            special_freqs=config.special_freqs,
        )
        self.blocks = nn.ModuleList(
            [
                ParallelBlock(
                    config.k, config.n_head, config.kv_groups, dropout=config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.k)
        self.l4 = nn.Linear(config.k, config.vocab_size + 1, bias=False)
        self.l4.weight = self.embedder.weight

    def forward(self, x):
        # x:  b, T
        r = self.embedder(x)  # (b, T, k)
        for block in self.blocks:
            r = block(r, self.rope, is_causal=True)  # (b, T, k)
        r = self.final_ln(r)  #  (b, T, k)
        r = self.l4(r)  #  (b, T, k)
        return r


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("mps")

    # # smaller config for a quick test
    # cfg = ModelConfig(
    #     name="test",
    #     vocab_size=71,
    #     block_size=64,
    #     n_head=8,
    #     n_layers=2,
    #     dropout=0.0,
    #     kv_groups=2,
    #     k=128,
    # )

    cfg = ModelConfig()

    model = ChessNet(cfg).to(device)
    model.eval()

    batch_size = 4
    seq_len = 16

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params = {total_params:,}")
    print(f"trainable params = {trainable_params:,}")

    import torch.optim as optim

    # random input tokens in [0, vocab_size-1]
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long).to(
        device
    )
    opt = optim.Adam(model.parameters(), lr=1e-4)

    opt.zero_grad()
    logits = model(x)  # (batch_size, seq_len, vocab_size)
    print("logits.shape =", logits.shape)

    # quick loss check
    targets = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long
    ).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    loss = loss_fn(logits.view(-1, cfg.vocab_size + 1), targets.view(-1))
    loss.backward()
    opt.step()
    print("loss =", loss.item())
