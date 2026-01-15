import torch
import math

import chess
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
        self.model.to(self.device)

        self.topk_list = [1, 3, 5, 10]
        self.topk_max = max(self.topk_list)

    @torch.no_grad()
    def metrics_by_pos_bucket_parity(
        self,
        batch_size=32,
        pos_bins=(0, 20, 40, 60, 80, 100, 10**9),
        max_games=None,
    ):
        """
        Compute negative log-likelihood and topk scores of true tokens:
        - per bins of plies
        - per parity of plies
        Returns a dictionary, with group labels.

        """
        pad = self.model.config.pad_index
        dl = build_dataloader(
            self.data_npz_path,
            batch_size=batch_size,
            padding_value=pad,
            max_length=500,
            shuffle=False,
            max_games=max_games,
        )
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
    def game_legality(self, seq: torch.Tensor):
        """
        Returns positions of illegal moves predicted in a batch.
        Removes 'end_token' prediction.
        """
        seq = seq.to(self.device)
        pad = self.model.config.pad_index

        x = seq[:, :-1]  # B, T
        y = seq[:, 1:]  # B, T

        B, T = y.shape

        logits = self.model(x)  # B, T, V
        logits = logits.clone()
        logits[:, :, self.encoder.end_token_id] = float("-inf")

        predicted_ids = logits.argmax(dim=-1)  # B, T

        illegals = torch.zeros_like(y, dtype=torch.bool)
        boards = [chess.Board() for _ in range(B)]

        # Get special token ids to skip
        start_id = self.encoder.start_token_id
        end_id = self.encoder.end_token_id

        for t in range(T):
            for b in range(B):
                true_id = int(y[b, t].item())
                if true_id == pad or true_id == start_id or true_id == end_id:
                    continue

                pid = int(predicted_ids[b, t].item())
                pmove = self.encoder.id_to_move(pid)
                illegals[b, t] = pmove is None or pmove not in boards[b].legal_moves

                tmove = self.encoder.id_to_move(true_id)
                boards[b].push(tmove)

        return illegals

    @torch.no_grad()
    def compute_legality_metrics(
        self,
        batch_size=32,
        pos_bins=(0, 20, 40, 60, 80, 100, 10**9),
        max_games=None,
    ):
        """
        Compute legality metrics across the dataset, bucketed by ply and parity.
        Returns the percentage of legal moves predicted (greedy argmax).
        """
        pad = self.model.config.pad_index
        dl = build_dataloader(
            self.data_npz_path,
            batch_size=batch_size,
            padding_value=pad,
            max_length=500,
            shuffle=False,
            max_games=max_games,
        )

        edges = torch.tensor(pos_bins, device=self.device, dtype=torch.long)
        nb = len(pos_bins) - 1
        G = nb * 2  # bucket x (even/odd)

        illegal_sum = torch.zeros(G, device=self.device, dtype=torch.float64)
        tok_sum = torch.zeros(G, device=self.device, dtype=torch.float64)

        for seq in tqdm(dl, desc="Checking legality"):
            illegals = self.game_legality(seq)  # [B, T]
            seq = seq.to(self.device)
            y = seq[:, 1:]
            mask = y != pad

            B, T = y.shape

            # "ply index"
            pos = torch.arange(1, T + 1, device=self.device).view(1, T).expand(B, T)

            bucket = torch.bucketize(pos, edges[1:-1], right=False)  # 0..nb-1
            group = bucket * 2 + (pos & 1)  # even/odd by position

            g = group[mask].reshape(-1)
            ill = illegals[mask].to(torch.float64).reshape(-1)

            illegal_sum.scatter_add_(0, g, ill)
            tok_sum.scatter_add_(0, g, torch.ones_like(ill))

        eps = 1e-12
        legality_per_group = 1.0 - (illegal_sum / (tok_sum + eps))

        total_moves = int(tok_sum.sum().item())
        total_illegal = int(illegal_sum.sum().item())
        total_legal = total_moves - total_illegal
        legality_rate = total_legal / total_moves if total_moves > 0 else 0.0

        return {
            "total_moves": total_moves,
            "legal_moves": total_legal,
            "illegal_moves": total_illegal,
            "legality_rate": legality_rate,
            "legality_per_group": legality_per_group.cpu().tolist(),
            "moves_per_group": tok_sum.cpu().tolist(),
            "group_labels": [
                f"[{pos_bins[b]},{pos_bins[b + 1]}) {'even' if p == 0 else 'odd'}"
                for b in range(nb)
                for p in (0, 1)
            ],
        }
