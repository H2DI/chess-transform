import torch
import torch.nn as nn

from .blocks import ReasoningNet


class TinyRecursiveChessModel(nn.Module):
    """
    TRM-style model:

      - Input: x_tokens (B, 69)  integer tokens for position
      - Internal variables:
          x_emb: (B, D)  question embedding
          y_emb:     (B, D)  embedded answer
          z:     (B, D)  latent reasoning state

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

        self.y_init = nn.Parameter(torch.zeros(1, 1, dim))
        self.z_init = nn.Parameter(torch.zeros(1, 1, dim))

        # Position token embedding (piece/empty + castling + turn)
        self.token_emb = nn.Embedding(token_vocab_size, dim)
        self.core = ReasoningNet(dim=dim, depth=core_depth, hidden_mult=hidden_mult)

        # Output head: y -> logits over moves
        self.out_head = nn.Linear(dim, num_moves, bias=False)
        self.q_head = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        y_emb: torch.Tensor = None,
        z: torch.Tensor = None,
    ):
        """
        x_tokens: (B, T)
        T = 69
        """
        device = x.device
        x_emb = self.token_emb(x)  # B, T, D

        if y_emb is None:
            y_emb = self.y_init.to(device)
        if z is None:
            z = self.z_init.to(device)

        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                for _ in range(self.L_cycles):
                    z = self.core(x_emb + z + y_emb)  # B, T, D
                y_emb = self.core(y_emb + z)

        for _ in range(self.L_cycles):
            z = self.core(x_emb + z + y_emb)  # B, T, D
        y_emb = self.core(y_emb + z)

        logits = self.out_head(y_emb[:, -1, :])  # B, V
        q_hat = self.q_head(y_emb[:, 0, :])  # B, 1
        return (y_emb.detach(), z.detach(), logits, q_hat)


class DeepSupervision(nn.Module):
    def __init__(self, base_model: nn.Module, opt, N=3, aux_loss_weight: float = 1):
        super().__init__()
        self.base_model = base_model
        self.opt = opt
        self.N = N
        self.y_init = None
        self.z_init = None
        self.aux_loss_weight = aux_loss_weight

    def forward(self, x: torch.Tensor, target_y):
        y_emb = self.y_init
        z = self.z_init
        losses = []
        for _ in range(self.N):
            self.opt.zero_grad()
            y_emb, z, logits, q_hat = self.base_model(x, y_emb, z)
            loss = nn.functional.cross_entropy(logits, target_y)

            yhat = logits.argmax(dim=-1)
            loss += (
                self.aux_loss_weight
                * nn.functional.binary_cross_entropy_with_logits(
                    q_hat.squeeze(), (yhat == target_y).float()
                )
            )
            loss.backward()
            losses.append(loss.item())
            self.opt.step()
        return losses
