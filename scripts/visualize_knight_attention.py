"""Plot attention rows at the plies where the white g1 knight moves in a PGN game.

- Finds the first game (or chosen index) in the PGN.
- Detects plies where the g1 knight moves.
- Runs the model and plots attention rows (one row per move) for selected layers/heads.
"""

import argparse
import math
from pathlib import Path

import chess
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
        help="Path to PGN file.",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=0,
        help="0-based index of game in PGN to analyze.",
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
        "--layers",
        type=str,
        default="0,-1",
        help="Comma-separated layer indices to plot (use -1 for last).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=0,
        help="Attention head index to visualize.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Truncate sequence to at most this many tokens for plotting.",
    )
    parser.add_argument(
        "--plot-rows",
        action="store_true",
        help="If set, plot 1xT attention rows for each knight ply instead of full TxT matrices.",
    )
    parser.add_argument(
        "--piece",
        type=str,
        default="g1",
        help="Initial square of the piece to follow (e.g., g1, e2, f8).",
    )
    parser.add_argument(
        "--single-plot",
        action="store_true",
        help="If set, produce one combined TxT attention plot (averaged across layers) and draw a vertical+horizontal line for each knight ply.",
    )
    parser.add_argument(
        "--all-heads",
        action="store_true",
        help="If set, run the plotting for every attention head and write one file per head.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="reports/knight_attention.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def load_game_at_index(pgn_path: Path, index: int) -> chess.pgn.Game:
    with open(pgn_path, "r") as pgn_file:
        for i in range(index + 1):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                raise ValueError(f"PGN ended before game index {index}")
        return game


def track_piece_plies(game: chess.pgn.Game, start_square_str: str):
    """Return ply indices where the piece that started on `start_square_str` moves.

    - `start_square_str` should be a string like 'g1' or 'e2'.
    - If the square is empty in the game's starting position, raises ValueError.
    - Stops tracking if the piece is captured or otherwise removed.
    """
    board = game.board()
    try:
        start_sq = chess.parse_square(start_square_str.lower())
    except Exception as e:
        raise ValueError(f"Invalid square string: {start_square_str}") from e

    piece = board.piece_at(start_sq)
    if piece is None:
        raise ValueError(f"No piece found on starting square {start_square_str}")

    tracked_square = start_sq
    tracked_color = piece.color
    tracked_type = piece.piece_type
    plies = []

    for ply_idx, move in enumerate(game.mainline_moves()):
        # If a move originates from the tracked square and matches piece identity,
        # record it and update tracked square.
        if move.from_square == tracked_square:
            # Confirm the mover matches the expected piece identity (in case of promotions)
            mover = board.piece_at(move.from_square)
            if mover is not None and mover.color == tracked_color:
                plies.append(ply_idx)
                tracked_square = move.to_square
        board.push(move)
        # After the move, check the tracked square still contains our piece
        if tracked_square is not None:
            p = board.piece_at(tracked_square)
            if p is None or p.color != tracked_color:
                # captured or replaced
                tracked_square = None
    return plies


def encode_game(game: chess.pgn.Game, encoder: MoveEncoder, max_tokens: int):
    seq = encoder.pgn_to_sequence(game)
    return np.array(seq[:max_tokens], dtype=np.int64)


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


def plot_knight_attention(
    seq,
    attn_maps,
    plies,
    layers,
    head,
    move_labels,
    out_path: Path,
    top_k: int = 3,
    plot_rows: bool = False,
    single_plot: bool = False,
):
    # Convert ply indices to token positions (start token at 0, move i at i+1)
    token_positions = [p + 1 for p in plies if p + 1 < len(seq)]
    # If there are no token positions (selected piece never moved in truncated sequence),
    # continue plotting but without drawing vertical/horizontal markers.

    # If requested, create a single combined attention plot (average across layers)
    if single_plot:
        # If multiple layers requested, produce one stacked subplot per layer.
        n_layers = len(layers)
        if n_layers > 1:
            # Choose a square size per subplot based on token count (so each matrix looks square)
            T = None
            mats = []
            for layer in layers:
                m = attn_maps[layer][head]
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().numpy()
                mats.append(np.asarray(m))
            T = mats[0].shape[0]
            pixels_per_token = 0.04
            square_size = max(6, min(16, T * pixels_per_token))
            fig_w = square_size
            fig_h = square_size * n_layers
            fig, axes = plt.subplots(n_layers, 1, figsize=(fig_w, fig_h), squeeze=False)
            for i, mat in enumerate(mats):
                ax = axes[i, 0]
                vmax = float(max(1e-9, mat.max()))
                im = ax.imshow(
                    mat,
                    aspect="equal",
                    origin="lower",
                    extent=[0, T, 0, T],
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax,
                )
                ax.set_title(f"Layer {layers[i]} | Head {head}")
                # draw lines for knight plies
                for tok_idx in token_positions:
                    ax.axvline(tok_idx, color="white", linewidth=1.0, linestyle="--")
                    ax.axhline(tok_idx, color="white", linewidth=1.0, linestyle="-")

                # sparse labels
                step = max(1, T // 32)
                xticks = list(range(0, T, step))
                xtick_labels = [move_labels[j] for j in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
                ax.set_yticks(xticks)
                ax.set_yticklabels(xtick_labels, fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200)
            print(f"[plot] saved {out_path}")
            return
        else:
            # single layer -> render the matrix directly (same as before)
            layer = layers[0]
            m = attn_maps[layer][head]
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            combined = np.asarray(m)

            T = combined.shape[0]
            pixels_per_token = 0.04
            square_size = max(6, min(16, T * pixels_per_token))
            fig_w = square_size
            fig_h = square_size
            fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
            vmax = float(max(1e-9, combined.max()))
            im = ax.imshow(
                combined,
                aspect="equal",
                origin="lower",
                extent=[0, T, 0, T],
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            ax.set_title(f"Layer {layer} | Head {head}")
            ax.set_xlabel("Key positions")
            ax.set_ylabel("Query positions")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # draw vertical and horizontal lines for each knight ply token position
            for tok_idx in token_positions:
                ax.axvline(tok_idx, color="white", linewidth=1.0, linestyle="--")
                ax.axhline(tok_idx, color="white", linewidth=1.0, linestyle="-")

            # Annotate x/y ticks sparsely with SAN labels
            step = max(1, T // 32)
            xticks = list(range(0, T, step))
            xtick_labels = [move_labels[i] for i in xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
            ax.set_yticks(xticks)
            ax.set_yticklabels(xtick_labels, fontsize=6)

            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200)
            print(f"[plot] saved {out_path}")
            return

    n_rows = len(token_positions)
    n_layers = len(layers)
    fig, axes = plt.subplots(
        n_rows,
        n_layers,
        figsize=(4 * n_layers, 4 * n_rows),
        squeeze=False,
    )

    for ri, tok_idx in enumerate(token_positions):
        for li, layer in enumerate(layers):
            attn_full = attn_maps[layer][head]  # (T, T)
            ax = axes[ri, li]

            # Normalize attention to a NumPy array so indexing/slicing uses numpy semantics
            if isinstance(attn_full, torch.Tensor):
                attn_full = attn_full.detach().cpu().numpy()
            attn_full = np.asarray(attn_full)

            if plot_rows:
                # Plot only the query row as a 1xT image
                row = attn_full[tok_idx][None, :]
                vmax = float(max(1e-9, row.max()))
                im = ax.imshow(
                    row,
                    aspect="auto",
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax,
                )
                ax.set_yticks([])
                ax.set_title(f"Ply {tok_idx - 1} | Layer {layer} | Head {head}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Top-k attended keys annotation (guard against small sequence lengths)
                k = max(0, min(top_k, row.shape[1]))
                if k > 0:
                    topk_idx = np.argsort(row[0])[-k:][::-1]
                    topk_labels = [
                        f"{i}:{move_labels[i]} ({row[0, i]:.2f})" for i in topk_idx
                    ]
                else:
                    topk_idx = []
                    topk_labels = []
                txt = "  \n".join(topk_labels)
                ax.text(
                    0.99,
                    0.01,
                    txt,
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="right",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5),
                )

                # Sparse x labels
                T = attn_full.shape[0]
                step = max(1, T // 16)
                xticks = list(range(0, T, step))
                xtick_labels = [move_labels[i] for i in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)
            else:
                # Plot full T x T attention matrix
                vmax = float(max(1e-9, attn_full.max()))
                im = ax.imshow(
                    attn_full,
                    aspect="auto",
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax,
                )

                # Mark the query row/column corresponding to the knight move token
                ax.axhline(tok_idx, color="white", linewidth=1.0)
                ax.axvline(tok_idx, color="white", linewidth=1.0, linestyle="--")

                ax.set_xlabel("Key positions")
                ax.set_ylabel("Query positions")
                ax.set_title(f"Ply {tok_idx - 1} | Layer {layer} | Head {head}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Top-k attended keys and their SAN labels
                row = attn_full[tok_idx]
                k = max(0, min(top_k, row.shape[0]))
                if k > 0:
                    topk_idx = np.argsort(row)[-k:][::-1]
                    topk_labels = [
                        f"{i}:{move_labels[i]} ({row[i]:.2f})" for i in topk_idx
                    ]
                else:
                    topk_idx = []
                    topk_labels = []
                txt = "  \n".join(topk_labels)
                ax.text(
                    0.99,
                    0.01,
                    txt,
                    transform=ax.transAxes,
                    fontsize=8,
                    va="bottom",
                    ha="right",
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5),
                )

                # Annotate x-axis with sparse SAN labels to avoid crowding
                T = attn_full.shape[0]
                step = max(1, T // 16)
                xticks = list(range(0, T, step))
                xtick_labels = [move_labels[i] for i in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=90, fontsize=6)

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

    game = load_game_at_index(Path(args.pgn), args.game_index)
    plies = track_piece_plies(game, args.piece)
    if not plies:
        print(
            f"[warn] The piece on {args.piece} never moved in this game — plotting without markers."
        )

    seq = encode_game(game, encoder, max_tokens=args.max_tokens)
    # Build SAN labels aligned with token indices: index 0 = <start>, index i>=1 = move i-1
    board = game.board()
    move_labels = ["<start>"]
    for move in game.mainline_moves():
        try:
            san = board.san(move)
        except Exception:
            san = move.uci()
        move_labels.append(san)
        board.push(move)
    # append end token label if present
    move_labels.append("<end>")
    # truncate/pad labels to match seq length
    if len(move_labels) < len(seq):
        move_labels += ["<pad>"] * (len(seq) - len(move_labels))
    else:
        move_labels = move_labels[: len(seq)]
    tokens = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    _, attn_maps = compute_attention_maps(model, tokens)

    layer_ids = [int(x) for x in args.layers.split(",") if x]

    # If requested, run for all heads
    if args.all_heads:
        # get number of heads from the first block's attention
        try:
            head_count = model.blocks[0].attn.heads
        except Exception:
            # fallback: assume 1 head
            head_count = 1

        base_out = Path(args.out)
        stem = base_out.stem
        suffix = base_out.suffix
        for h in range(head_count):
            out_path = base_out.with_name(f"{stem}_h{h}{suffix}")
            plot_knight_attention(
                seq,
                attn_maps,
                plies,
                layers=layer_ids,
                head=h,
                move_labels=move_labels,
                out_path=out_path,
                top_k=3,
                plot_rows=args.plot_rows,
                single_plot=args.single_plot,
            )
    else:
        plot_knight_attention(
            seq,
            attn_maps,
            plies,
            layers=layer_ids,
            head=args.head,
            move_labels=move_labels,
            out_path=Path(args.out),
            top_k=3,
            plot_rows=args.plot_rows,
            single_plot=args.single_plot,
        )


if __name__ == "__main__":
    main()
