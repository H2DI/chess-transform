import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import time
import matplotlib.pyplot as plt


# Trial of KV cache before actual implementation.


class MHA(nn.Module):
    def __init__(self, k_in, kh, h):
        super().__init__()
        self.k_in = k_in
        self.h = h
        self.kh = kh
        self.Q = nn.Linear(k_in, h * kh, bias=False)
        self.KV = nn.Linear(k_in, 2 * h * kh, bias=False)

        self.lout = nn.Linear(h * kh, k_in, bias=False)
        self.Kcache = None
        self.Vcache = None

    def forward(self, x: torch.Tensor, kv_cache=False, fill_cache=False, mask=None):
        if not (kv_cache):
            b, T, k_in = x.shape
            assert k_in == self.k_in
            Q = self.Q(x)
            K, V = torch.split(self.KV(x), [self.h * self.kh, self.h * self.kh], dim=-1)

            Q = Q.reshape(b, T, self.h, self.kh).transpose(1, 2)
            K = K.reshape(b, T, self.h, self.kh).transpose(1, 2)
            V = V.reshape(b, T, self.h, self.kh).transpose(1, 2)  # b, h, T, kh

            if fill_cache:
                self.Kcache = K
                self.Vcache = V

            scores = Q @ K.transpose(-1, -2) / math.sqrt(self.kh)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-inf"))
            attention = F.softmax(scores, dim=-1) @ V
            # b, h, T, kh

            attention = attention.transpose(1, 2).reshape(b, T, self.h * self.kh)
            # b, T, h * kh
            return self.lout(attention)  # b, T, k
        else:
            last_x = x  # b, 1, k
            b, _, k = x.shape

            last_k, last_v = torch.split(self.KV(last_x), self.h * self.kh, dim=-1)
            last_k = last_k.reshape(b, 1, self.h, self.kh).transpose(1, 2)
            last_v = last_v.reshape(b, 1, self.h, self.kh).transpose(1, 2)
            if self.Kcache is None:
                self.Kcache = last_k
                self.Vcache = last_v
            else:
                self.Kcache = torch.cat([self.Kcache, last_k], dim=2)
                self.Vcache = torch.cat([self.Vcache, last_v], dim=2)

            Q = self.Q(last_x).reshape(b, 1, self.h, self.kh).transpose(1, 2)

            attention = (
                F.softmax(
                    Q @ self.Kcache.transpose(-1, -2) / math.sqrt(self.kh), dim=-1
                )
                @ self.Vcache
            )
            # b, h, 1, kh
            attention = attention.transpose(1, 2).reshape(b, 1, self.h * self.kh)
            return self.lout(attention)  # b, 1, k

    def reset_cache(self):
        self.Kcache = None
        self.Vcache = None


class TransformerBlock(nn.Module):
    def __init__(self, k_in, kh, h, mlp_hidden, p):
        super().__init__()
        self.attention = MHA(k_in, kh, h)
        self.mlp = nn.Sequential(
            nn.Linear(k_in, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, k_in)
        )
        self.dropout = nn.Dropout(p)
        self.norm1 = nn.LayerNorm(k_in)
        self.norm2 = nn.LayerNorm(k_in)

    def forward(self, x, kv_cache=False, fill_cache=False, mask=None):
        attended = self.attention(
            self.norm1(x), kv_cache=kv_cache, fill_cache=fill_cache, mask=mask
        )
        x = x + attended
        x = self.dropout(x)
        fedforward = self.mlp(self.norm2(x))
        return fedforward + x

    def reset_cache(self):
        self.attention.reset_cache()


class PositionalEncoder(nn.Module):
    def __init__(self, k, block_size):
        super().__init__()
        self.block_size = block_size
        self.k = k
        freqs = 2 * math.pi * torch.arange(block_size).unsqueeze(1)
        freq2 = (1 / (1 + 100 * torch.arange(k))).unsqueeze(0)
        pe_mat = torch.cos(freqs * freq2)
        self.register_buffer("pe_mat", pe_mat)

    def forward(self, x, T0=0):
        _, T, _ = x.shape

        return x + self.pe_mat[T0 : T0 + T, :].unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, vocab, k, kh, h, hidden, p, block_size, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab, k)
        self.layers = nn.ModuleList(
            [TransformerBlock(k, kh, h, hidden, p) for _ in range(n_layers)]
        )
        self.pe = PositionalEncoder(k, block_size)
        self.deembedding = nn.Linear(k, vocab, bias=False)
        self.deembedding.weight = self.embedding.weight

    def forward(self, x):
        x_embed = self.embedding(x)
        x_pos = self.pe(x_embed)
        T = x.size(1)
        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
        )
        for layer in self.layers:
            x_pos = layer(x_pos, mask=mask)
        return self.deembedding(x_pos)

    def forward_with_cache(self, x, t):
        # x: b, 1
        x_embed = self.embedding(x)  # b, 1, k
        x_pos = self.pe(x_embed, T0=t)  # b, 1, k
        for layer in self.layers:
            x_pos = layer(x_pos, kv_cache=True)  # b, 1, k
        return self.deembedding(x_pos)  # b, 1, v

    def fill_cache(self, prefix):
        prefix_embedding = self.embedding(prefix)
        x_pos = self.pe(prefix_embedding)
        for layer in self.layers:
            x_pos = layer(x_pos, fill_cache=True)
        return self.deembedding(x_pos)

    @torch.no_grad()
    def generate_with_kvcache(self, T, prefix=None):
        self.reset_cache()
        if prefix is None:
            start = 0
            next_token = torch.tensor([[0]])  # 1, T
            out = next_token
        else:
            start = prefix.size(1)
            logits = self.fill_cache(prefix)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            out = torch.cat([prefix, next_token], dim=-1)
        for t in range(T):
            if prefix is not None and (t == T - 1):
                break
            logits = self.forward_with_cache(next_token, start + t)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            out = torch.cat([out, next_token], dim=-1)
        return out

    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()

    @torch.no_grad()
    def generate(self, T, prefix=None):
        if prefix is None:
            x = torch.tensor([[0]])
        else:
            x = prefix
        for _ in range(T):
            logits = self.forward(x)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            x = torch.cat([x, next_token], dim=-1)
        return x


k_in, kh, h = 4, 4, 2
k = k_in
b, T = 2, 4
hidden = 8
p = 0

v = 120
x = torch.randn([b, T, k_in])
layer = MHA(k_in, kh, h)


block = TransformerBlock(k_in, kh, h, hidden, 0.1)

model = Transformer(v, k, kh, h, hidden, p, block_size=10000, n_layers=2)

prefix = torch.randint(0, v, size=[2, 32])

_ = model.generate_with_kvcache(1)
t0 = time.perf_counter()
x_true = model.generate_with_kvcache(300, prefix=prefix)
t1 = time.perf_counter()
dur_true = t1 - t0

_ = model.generate(1)
t0 = time.perf_counter()
x_false = model.generate(300, prefix=prefix)
t1 = time.perf_counter()
dur_false = t1 - t0

print(x_true.shape)
print(x_false.shape)

print((x_true.isclose(x_false)))
print((x_true == x_false).all())

print(f"kv_cache=True:  {dur_true:.6f}s")
print(f"kv_cache=False: {dur_false:.6f}s")
print(f"difference (false - true): {dur_false - dur_true:.6f}s")

### Dual generation

# A = Transformer(v, k, kh, h, hidden, p, block_size=1000, n_layers=3)
# B = Transformer(v, k, kh, h, hidden, p, block_size=1000, n_layers=3)

# prefix = torch.randint(0, v, size=[32, 32])


# A.fill_cache(prefix)
# start = prefix.size(1)
# logits = B.fill_cache(prefix)
# next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
# out1 = torch.cat([prefix, next_token], dim=-1)

# for t in range(100):
#     if t // 2:
#         logits = A.forward_with_cache(next_token, start + 2 * t)
#         next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
#         out1 = torch.cat([out1, next_token], dim=-1)
#         B.forward_with_cache(next_token, start + 2 * t + 1)  # update cache
#     else:
#         logits = B.forward_with_cache(next_token, start + 2 * t)
#         next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
#         out1 = torch.cat([out1, next_token], dim=-1)
#         A.forward_with_cache(next_token, start + 2 * t + 1)  # update cache

# print(out1.shape)

# A.reset_cache()
# B.reset_cache()

# A.fill_cache(prefix)
# start = prefix.size(1)
# logits = B.fill_cache(prefix)
# next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
# out2 = torch.cat([prefix, next_token], dim=-1)

# for t in range(100):
#     if t // 2:
#         logits = A.forward_with_cache(next_token, start + 2 * t)
#         next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
#         out2 = torch.cat([out2, next_token], dim=-1)
#     else:
#         logits = B.forward_with_cache(next_token, start + 2 * t)
#         next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
#         out2 = torch.cat([out2, next_token], dim=-1)

# print((out1 == out2).all())

# m = PositionalEncoder(100, 200)
# plt.imshow(m.pe_mat)
# plt.show()
