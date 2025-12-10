"""Visualize attention maps for the gamba_rossa model on Ivanchuk PGN games.

Outputs a heatmap per selected layer/head for a chosen ply in each sampled game.
"""

import argparse
import math
from pathlib import Path

import chess.pgn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from chess_seq import MoveEncoder
from chess_seq import load_model

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn",
        type=str,
        default="data/gms_pgns/Ivanchuk.pgn",
        help="Path to PGN file to visualize.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gamba_rossa",
        help="Model name under checkpoints/.",
    )
    parser.add_argument(
        "--special-name",
        type=str,
        default="final",
        help="Checkpoint filename inside checkpoints/<model-name>/.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device string (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=2,
        help="How many games to visualize from the PGN.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,-1",
        help="Comma-separated layer indices to plot (use -1 for last).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Attention head index to visualize in each layer.",
    )
    parser.add_argument(
        "--focus-ply",
        type=int,
        default=-1,
        help="Ply index to visualize (default last ply).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="Truncate sequences to at most this many tokens for plotting.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/ivanchuk_attention.png",
        help="Output PNG path for the attention heatmaps.",
    )
    return parser.parse_args()


def load_games(pgn_path: Path, encoder: MoveEncoder, limit: int, max_tokens: int):
    sequences = []
    with open(pgn_path, "r") as pgn_file:
        pbar = tqdm(total=limit, desc="Reading PGN", unit="game")
        while len(sequences) < limit:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            seq = encoder.pgn_to_sequence(game)
            sequences.append(np.array(seq[:max_tokens], dtype=np.int64))
            pbar.update(1)
        pbar.close()
    if not sequences:
        raise ValueError(f"No games found in {pgn_path}")
    return sequences


def compute_attention_maps(model, token_ids: torch.Tensor):
    """Run a forward pass and capture attention maps per layer.

    Returns logits and a list of attention tensors shaped (heads, T, T).
    """

    model.eval()
    rope = model.rope
    r = model.embedder(token_ids)  # (1, T, k)
    attn_maps = []

    for block in model.blocks:
        x = block.norm1(r)
        attn = block.attn
        b, T, _ = x.shape
        h, groups = attn.heads, attn.groups
        hg = h // groups
        kh = attn.head_dim

        Q = attn.q(x)
        K, V = torch.split(attn.kv(x), [hg * kh, hg * kh], dim=-1)

        Q = Q.view(b, T, h, kh).transpose(1, 2)
        K = K.reshape(b, T, hg, kh).transpose(1, 2).repeat_interleave(groups, dim=1)
        V = V.reshape(b, T, hg, kh).transpose(1, 2).repeat_interleave(groups, dim=1)

        Q = attn.qnorm(Q)
        K = attn.knorm(K)

        Q = rope(Q)
        K = rope(K)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(kh)
        causal_mask = torch.triu(
            torch.ones(T, T, device=token_ids.device), diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_prob = torch.softmax(attn_scores, dim=-1)  # (b, h, T, T)
        attn_maps.append(attn_prob.detach().cpu()[0])

        context = torch.matmul(attn_prob, V)  # (b, h, T, kh)
        context = context.transpose(1, 2).reshape(b, T, h * kh)
        out = attn.unify_heads(context)
        r = r + out
        r = r + block.dropout(block.ff(block.norm2(r)))

    r = model.final_ln(r)
    logits = model.l4(r)
    return logits, attn_maps


def plot_attention(
    sequences, attn_maps_list, layers, head, focus_ply, encoder, out_path: Path
):
    n_games = len(sequences)
    n_layers = len(layers)
    fig, axes = plt.subplots(
        n_games, n_layers, figsize=(4 * n_layers, 4 * n_games), squeeze=False
    )

    for gi, (seq, attn_maps) in enumerate(zip(sequences, attn_maps_list)):
        T = len(seq)
        focus_idx = focus_ply if focus_ply >= 0 else T - 1
        focus_idx = max(0, min(focus_idx, T - 1))
        for li, layer in enumerate(layers):
            attn = attn_maps[layer][head]  # (T, T)
            ax = axes[gi, li]
            row = attn[focus_idx][None, :]  # (1, T) so imshow gets 2D
            vmax = max(1e-9, row.max())
            im = ax.imshow(
                row, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=vmax
            )
            ax.set_yticks([])
            ax.set_title(
                f"Game {gi + 1} | Layer {layer} | Head {head}\nFocus ply {focus_idx}"
            )
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query ply")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"[plot] saved {out_path}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, model_config, _ = load_model(
        model_name=args.model_name,
        special_name=args.special_name,
    )
    model = model.to(device)

    encoder = MoveEncoder()
    encoder.load(model_config.encoder_path)

    sequences = load_games(
        Path(args.pgn), encoder, limit=args.num_games, max_tokens=args.max_tokens
    )

    layer_ids = [int(x) for x in args.layers.split(",") if x]
    attn_maps_list = []
    for seq in tqdm(sequences, desc="Computing attention", unit="game"):
        tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
        _, attn_maps = compute_attention_maps(model, tokens)
        attn_maps_list.append(attn_maps)

    plot_attention(
        sequences,
        attn_maps_list,
        layers=layer_ids,
        head=args.head,
        focus_ply=args.focus_ply,
        encoder=encoder,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    main()
