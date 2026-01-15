import torch
import math

import chess
import torch.nn.functional as F
from tqdm import tqdm

from ..encoder import MoveEncoder
from ..models import ChessNet
from ..datasets.datasets import build_dataloader


class Evaluator:
    def __init__(
        self,
        model: ChessNet,
        encoder: MoveEncoder,
        npz_path: str,
        device=None,
    ):
        self.model = model
        self.encoder = encoder
        self.device = device or torch.device("cpu")
        self.data_npz_path = npz_path
        self.model.to(device)

        self.topk_list = [1, 3, 5, 10]
        self.topk_max = max(self.topk_list)

    def compute_prediction_metrics_by_pos_bucket_parity(
        self,
        batch_size=32,
        pos_bins=(0, 20, 40, 60, 80, 100, 10**9),
        max_games=None,
    ):
        # same dataloader behavior as your original code
        dl = build_dataloader(
            self.data_npz_path,
            batch_size=batch_size,
            padding_value=4610,
            max_length=500,
            shuffle=False,
            max_games=max_games,
        )
        pad = self.model.config.pad_index
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad, reduction="none")

        topk_list = self.topk_list
        kmax = self.topk_max

        edges = torch.tensor(pos_bins, device=self.device, dtype=torch.long)
        nb = len(pos_bins) - 1
        G = nb * 2  # bucket x (even/odd)

        loss_sum = torch.zeros(G, device=self.device, dtype=torch.float64)
        tok_sum = torch.zeros(G, device=self.device, dtype=torch.float64)
        hit_sum = {
            k: torch.zeros(G, device=self.device, dtype=torch.float64)
            for k in topk_list
        }

        for seq in tqdm(dl, desc="Evaluating batches"):
            seq = seq.to(self.device)
            x = seq[:, :-1]
            y = seq[:, 1:]
            logits = self.model(x)

            mask = y != pad

            B, T = y.shape

            # "ply index"
            pos = torch.arange(1, T + 1, device=self.device).view(1, T).expand(B, T)

            bucket = torch.bucketize(pos, edges[1:-1], right=False)  # 0..nb-1
            group = bucket * 2 + (pos & 1)  # even/odd by position

            g = group[mask].reshape(-1)

            # loss per token
            per_tok = criterion(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )  # [B*T]
            per_tok = per_tok.view(B, T)  # [B,T]

            nll = per_tok[mask].to(torch.float64).reshape(-1)  # [N]

            loss_sum.scatter_add_(0, g, nll)
            tok_sum.scatter_add_(0, g, torch.ones_like(nll))

            # top-k hits
            topks = logits.topk(kmax, dim=-1).indices  # [B,T,K]
            hit = topks == y.unsqueeze(-1)  # [B,T,K]
            for k in topk_list:
                hk = (
                    hit[:, :, :k].any(dim=-1)[mask].to(torch.float64).reshape(-1)
                )  # [N]
                hit_sum[k].scatter_add_(0, g, hk)

        eps = 1e-12
        avg_nll = loss_sum / (tok_sum + eps)
        ppl = torch.exp(avg_nll)

        return {
            "avg_nll_per_group": avg_nll.cpu().tolist(),
            "perplexity_per_group": ppl.cpu().tolist(),
            "tokens_per_group": tok_sum.cpu().tolist(),
            "topk_per_group": {
                k: (hit_sum[k] / (tok_sum + eps)).cpu().tolist() for k in topk_list
            },
            "overall": {
                "avg_nll": float((loss_sum.sum() / (tok_sum.sum() + eps)).item()),
                "perplexity": float(
                    math.exp((loss_sum.sum() / (tok_sum.sum() + eps)).item())
                ),
                "tokens": float(tok_sum.sum().item()),
                "topk_scores": {
                    k: float((hit_sum[k].sum() / (tok_sum.sum() + eps)).item())
                    for k in topk_list
                },
            },
            # group order is bucket0-even, bucket0-odd, bucket1-even, bucket1-odd, ...
            "group_labels": [
                f"[{pos_bins[b]},{pos_bins[b + 1]}) {'even' if p == 0 else 'odd'}"
                for b in range(nb)
                for p in (0, 1)
            ],
        }

    @torch.no_grad()
    def single_game_legality(self, seq):
        """
        Returns positions of illegal moves predicted
        """
        T = len(seq)
        illegals = torch.zeros(T, dtyp=torch.bool)
        board = chess.Board()
        for t in range(T - 1):
            current_game = seq[:t]

            # predicted next move
            logits = self.model(current_game)
            logits[self.encoder.end_token_id] = float("-inf")
            predicted_id = torch.argmax(logits, dim=-1)
            predicted_move = self.encoder.id_to_move(predicted_id)

            illegals[t] = predicted_move in board.legal_moves

            # true next move
            move = self.encoder.id_to_move(seq[t + 1])
            board.push(move)
        return illegals.cpu().numpy()


# report in a json
# compute perplexity


# todo:

# 6000 games
# compute perplexity
# compute top 1, top 3, top 5, top 10
# separate black and white
# plot top 5 accuracy per ply bucket


# compute illegal moves in top1 in next 10 plies after 11, 22, 33 plies
