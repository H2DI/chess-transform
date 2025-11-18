import math
import torch
import torch.nn as nn

from .blocks import ReasoningNet


class TinyRecursiveChessModel(nn.Module):
    """
    TRM-style model:

      - Input: x_tokens (B, 69)  integer tokens for position
      - Internal latents:
          x_emb: (B, D)  question embedding
          y:     (B, D)  embedded answer
          z:     (B, D)  latent reasoning state

      - Recursion:
          For h in 1..H_cycles:
              For l in 1..L_cycles:   # latent refinement
                  z = core_z( concat(x_emb, y, z) )
              y = core_y( concat(y, z) )   # answer refinement
              logits_h = head(y)

      - Output:
          logits at each outer step (for deep supervision) or final step only.
    """

    def __init__(
        self,
        token_vocab_size: int,
        num_moves: int,
        dim: int = 256,
        seq_len: int = 69,
        H_cycles: int = 3,
        L_cycles: int = 4,
        core_depth: int = 2,
        hidden_mult: int = 4,
    ):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        # Position token embedding (piece/empty + castling + turn)
        self.token_emb = nn.Embedding(token_vocab_size, dim)

        # Learned positional embedding over 69 tokens
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim) / math.sqrt(dim))

        # Initial answer embedding y0
        self.y0 = nn.Parameter(torch.zeros(dim))

        # Shared tiny reasoning network
        self.core = ReasoningNet(dim=dim, depth=core_depth, hidden_mult=hidden_mult)

        # Linear adaptors for z- and y-updates
        self.z_in = nn.Linear(3 * dim, dim, bias=False)  # [x_emb, y, z] -> dim
        self.y_in = nn.Linear(2 * dim, dim, bias=False)  # [y, z] -> dim

        # Output head: y -> logits over moves
        self.head = nn.Linear(dim, num_moves, bias=False)

    # ------------------------
    # Embedding of x (position)
    # ------------------------
    def embed_x(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        x_tokens: (B, seq_len) with values in [0, token_vocab_size)
        returns x_emb: (B, D)
        """
        B, T = x_tokens.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got {T}"

        tok = self.token_emb(x_tokens)  # (B, T, D)
        pos = self.pos_emb.unsqueeze(0)  # (1, T, D)
        h = tok + pos
        x_emb = h.mean(dim=1)  # mean-pool -> (B, D)
        return x_emb

    # ------------------------
    # Forward
    # ------------------------
    def forward(self, x_tokens: torch.Tensor, return_all_steps: bool = False):
        """
        x_tokens: (B, 69)

        If return_all_steps:
            returns logits_all: (H_cycles, B, num_moves)
        else:
            returns logits_final: (B, num_moves)
        """
        device = x_tokens.device
        B = x_tokens.size(0)

        x_emb = self.embed_x(x_tokens)  # (B, D)

        # init y and z
        y = self.y0.unsqueeze(0).expand(B, -1)  # (B, D)
        z = torch.zeros_like(y, device=device)  # (B, D)

        logits_list = []

        for _ in range(self.H_cycles):
            # latent refinement (L cycles)
            for _ in range(self.L_cycles):
                inp_z = torch.cat([x_emb, y, z], dim=-1)  # (B, 3D)
                z = self.z_in(inp_z)  # (B, D)
                z = self.core(z)  # (B, D)

            # answer refinement
            inp_y = torch.cat([y, z], dim=-1)  # (B, 2D)
            y = self.y_in(inp_y)  # (B, D)
            y = self.core(y)  # (B, D)

            logits = self.head(y)  # (B, V)
            logits_list.append(logits)

        if return_all_steps:
            return torch.stack(logits_list, dim=0)  # (H, B, V)
        else:
            return logits_list[-1]  # (B, V)
