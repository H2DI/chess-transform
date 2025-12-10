"""Evaluate a saved model on a PGN file and plot <end> probabilities.

Steps:
- encode PGN games to NPZ with token ids
- compute loss on the encoded games
- visualize the predicted probability of the <end> token across plies
"""

import argparse
import math
import sys
from pathlib import Path

import chess.pgn
import matplotlib
import numpy as np
import torch
from tqdm.auto import tqdm

from chess_seq import MoveEncoder
from chess_seq import build_dataloader
from chess_seq import load_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn",
        type=str,
        default="data/gms_pgns/Ivanchuk.pgn",
        help="Path to input PGN file.",
    )
    parser.add_argument(
        "--out-npz",
        type=str,
        default="data/gms_pgns/ivanchuk_encoded.npz",
        help="Where to write the encoded NPZ file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gamba_rossa",
        help="Model name as saved under checkpoints/.",
    )
    parser.add_argument(
        "--special-name",
        type=str,
        default="final",
        help="Special checkpoint name (e.g., final.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string for torch (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-games-plot",
        type=int,
        default=3,
        help="How many games to include in the <end> probability plot.",
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default="reports/ivanchuk_end_token_probs.png",
        help="Where to save the plot image.",
    )
    return parser.parse_args()


def _resolve_path(path_like: Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_encoder(encoder_path: str) -> MoveEncoder:
    path = _resolve_path(Path(encoder_path))
    encoder = MoveEncoder()
    encoder.load(str(path))
    return encoder


def convert_pgn_to_npz(pgn_path: Path, npz_out: Path, encoder: MoveEncoder):
    game_ids = []
    tokens = []

    pgn_path = _resolve_path(pgn_path)
    npz_out = _resolve_path(npz_out)

    with open(pgn_path, "r") as pgn_file:
        pbar = tqdm(desc="Encoding PGN", unit="game")
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            seq = encoder.pgn_to_sequence(game)
            tokens.append(np.array(seq, dtype=np.int32))
            game_ids.append(len(game_ids))
            pbar.update(1)
        pbar.close()

    if not tokens:
        raise ValueError(f"No games found in {pgn_path}")

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_out,
        game_ids=np.array(game_ids, dtype=np.int32),
        tokens=np.array(tokens, dtype=object),
    )
    print(f"[encode] saved {len(tokens)} games to {npz_out}")
    return tokens


def compute_loss(model, dataloader, pad_id: int, device: torch.device):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += float(loss.item())
            total_tokens += int((targets != pad_id).sum().item())

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def end_token_probabilities(model, sequence, end_token_id: int, device: torch.device):
    seq_tensor = torch.tensor(sequence[:-1], dtype=torch.long, device=device).unsqueeze(
        0
    )
    with torch.no_grad():
        logits = model(seq_tensor)[0]
        probs = torch.softmax(logits, dim=-1)[:, end_token_id].cpu().numpy()
    move_numbers = np.arange(1, len(sequence))
    return probs, move_numbers


def plot_end_token_probabilities(prob_sets, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(prob_sets)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for idx, (probs, move_numbers) in enumerate(prob_sets):
        even_mask = move_numbers % 2 == 0
        odd_mask = ~even_mask
        ax = axes[idx]
        ax.plot(
            move_numbers, probs, color="black", alpha=0.6, linewidth=1.0, label="all"
        )
        ax.scatter(
            move_numbers[even_mask],
            probs[even_mask],
            color="#1f77b4",
            s=12,
            label="even plies",
        )
        ax.scatter(
            move_numbers[odd_mask],
            probs[odd_mask],
            color="#ff7f0e",
            s=12,
            label="odd plies",
        )
        ax.set_ylabel("P(<end>)")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Game {idx + 1} | plies={len(move_numbers)}")
    axes[-1].set_xlabel("Ply")
    axes[0].legend()
    fig.tight_layout()
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
    model.eval()
    encoder = load_encoder(model_config.encoder_path)

    pgn_path = Path(args.pgn)
    npz_out = _resolve_path(Path(args.out_npz))
    plot_out = _resolve_path(Path(args.plot_out))

    tokens = convert_pgn_to_npz(pgn_path, npz_out, encoder)

    dataloader = build_dataloader(
        npz_path=str(npz_out),
        batch_size=args.batch_size,
        padding_value=model_config.pad_index,
        max_length=model_config.block_size,
        shuffle=False,
    )

    avg_loss, ppl = compute_loss(model, dataloader, model_config.pad_index, device)
    print(f"[eval] avg per-token loss={avg_loss:.4f} | ppl={ppl:.2f}")

    prob_sets = []
    for i, seq in enumerate(tokens):
        if i >= args.num_games_plot:
            break
        trimmed = seq[: model_config.block_size]
        probs, move_numbers = end_token_probabilities(
            model, trimmed, encoder.end_token_id, device
        )
        prob_sets.append((probs, move_numbers))

    if prob_sets:
        plot_end_token_probabilities(prob_sets, plot_out)
    else:
        print("[plot] no games available for plotting")


if __name__ == "__main__":
    main()
