import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import ModelConfig


class RoPE(nn.Module):
    def __init__(self, d_model, block_size, special_freqs=None):
        super().__init__()
        assert d_model % 2 == 0
        special_freqs = special_freqs or []

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
        device = device or self.inv_freq.device
        positions = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(1)
        # N, 1
        return positions @ self.inv_freq  # N, k //2

    def _recompute_cossin(self, N, device):
        angles = self._get_angles(N, device)
        return torch.cos(angles), torch.sin(angles)


class GQARope(nn.Module):
    """
    Group query attention with QK normalization
    """

    def __init__(self, k, head_dim, heads=16, groups=2, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.groups = groups
        self.dim = heads * head_dim
        self.qnorm = nn.RMSNorm(self.head_dim)
        self.knorm = nn.RMSNorm(self.head_dim)

        self.q = nn.Linear(k, self.heads * self.head_dim, bias=False)
        self.kv = nn.Linear(k, 2 * heads // groups * self.head_dim, bias=False)
        self.unify_heads = nn.Linear(heads * self.head_dim, k, bias=False)

        self.dropout_p = dropout

    def forward(self, x, rope=None, is_causal=True):
        b, T, _ = x.shape
        h, groups = self.heads, self.groups
        hg = h // groups  # kv_heads = h / g
        kh = self.head_dim

        # Q: b, T, h * kh
        # K, V: b, T, (h/g) * kh
        Q = self.q(x)
        K, V = torch.split(
            self.kv(x),
            [hg * kh, hg * kh],
            dim=-1,
        )

        # Q: b, h, T, k/h
        Q = Q.view(b, T, h, kh).transpose(1, 2)

        # K, V: b, h/g, T, k/h
        K = K.reshape(b, T, hg, kh).transpose(1, 2).repeat_interleave(groups, dim=1)
        V = V.reshape(b, T, hg, kh).transpose(1, 2).repeat_interleave(groups, dim=1)

        # b, h, T, k/h
        Q = self.qnorm(Q)
        K = self.knorm(K)

        if rope is not None:
            Q, K = rope(Q), rope(K)

        out = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )
        # b, h, T, kh

        # out: b, T, k
        out = out.transpose(1, 2).reshape(b, T, h * kh)
        return self.unify_heads(out)


class SwiGLUBlock(nn.Module):
    # Replace MLP in Transformer block

    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim or 3 * dim
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w_down(self.w_up(x) * F.silu(self.w_gate(x)))


class DecoderLayer(nn.Module):
    def __init__(self, dim, head_dim, n_heads, groups, dropout=0.0, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 3 * dim
        self.norm1 = nn.RMSNorm(dim)
        self.attn = GQARope(dim, head_dim, n_heads, groups, dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.ff = SwiGLUBlock(dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope, is_causal=True):
        attn_out = self.attn(self.norm1(x), rope=rope, is_causal=is_causal)
        x = x + attn_out
        ff_out = self.ff(self.norm2(x))
        return x + self.dropout(ff_out)


class ChessNet(nn.Module):
    def __init__(self, config=ModelConfig):
        """
        Copy of Qwen3
        """
        super().__init__()
        self.config = config
        self.embedder = nn.Embedding(
            config.vocab_size, config.k, padding_idx=config.pad_index
        )
        self.rope = RoPE(
            config.head_dim,
            config.block_size,
            special_freqs=config.special_freqs,
        )
        self.blocks = nn.ModuleList(
            [
                DecoderLayer(
                    config.k,
                    config.head_dim,
                    config.n_head,
                    config.kv_groups,
                    dropout=config.dropout,
                    hidden_dim=config.ff_expansion * config.k,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_ln = nn.RMSNorm(config.k)
        self.l4 = nn.Linear(config.k, config.vocab_size, bias=False)
        self.l4.weight = self.embedder.weight

    def forward(self, x):
        # x:  b, T
        r = self.embedder(x)  # (b, T, k)
        for block in self.blocks:
            r = block(r, rope=self.rope, is_causal=True)  # (b, T, k)
        r = self.final_ln(r)  #  (b, T, k)
        r = self.l4(r)  #  (b, T, k)
        return r


if __name__ == "__main__":
    import torch.optim as optim

    torch.manual_seed(0)
    device = torch.device("cpu")

    # smaller config for a quick test
    cfg = ModelConfig(
        name="test",
        vocab_size=4611,
        block_size=64,
        n_head=8,
        n_layers=2,
        dropout=0.0,
        kv_groups=2,
        k=128,
    )

    cfg = ModelConfig()
    pad_id = cfg.vocab_size - 1

    model = ChessNet(cfg).to(device)
    model.eval()

    batch_size = 2
    seq_len = 16

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params = {total_params:,}")
    print(f"trainable params = {trainable_params:,}")

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
    loss = loss_fn(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    opt.step()
    print("loss =", loss.item())
