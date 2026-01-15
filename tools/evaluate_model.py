#!/usr/bin/env python3
"""
Evaluate a chess model on test data and write a report.

Example:
    python tools/evaluate_model.py \
        --data data/carlsen_games/shard_00000.npz \
        --output reports/carlsen_evaluation.json \
        --max-games 500
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Ensure chess_seq is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_seq.utils.save_and_load import load_model_from_safetensors
from chess_seq.evaluation.test_loss import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess model on test data")
    parser.add_argument(
        "--model-name",
        type=str,
        default="gamba_rossa",
        help="Model name (folder in checkpoints/)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test data NPZ file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report path (default: reports/<data_basename>_eval.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to evaluate (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Validate data path
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    # Load model and encoder from safetensors
    print(f"Loading model {args.model_name}...")
    model, config, encoder = load_model_from_safetensors(args.model_name)
    model.to(device)
    model.eval()

    # Create evaluator
    max_games_str = f" (max {args.max_games} games)" if args.max_games else ""
    print(f"Evaluating on {args.data}{max_games_str}...")
    evaluator = Evaluator(
        model=model,
        encoder=encoder,
        npz_path=args.data,
        device=device,
    )

    # Run evaluation
    with torch.no_grad():
        metrics = evaluator.metrics_by_pos_bucket_parity(
            batch_size=args.batch_size,
            max_games=args.max_games,
        )
        legality_metrics = evaluator.compute_legality_metrics(
            batch_size=args.batch_size,
            max_games=args.max_games,
        )

    # Add metadata to report
    report = {
        "metadata": {
            "model_name": args.model_name,
            "data": args.data,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_games": args.max_games,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": metrics,
        "legality": legality_metrics,
    }

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        data_basename = Path(args.data).stem
        output_path = f"reports/{data_basename}_eval.json"

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write report
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")

    # Print summary
    overall = metrics["overall"]
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total tokens evaluated: {overall['tokens']:,.0f}")
    print(f"Average NLL: {overall['avg_nll']:.4f}")
    print(f"Perplexity: {overall['perplexity']:.4f}")
    print("\nTop-k Accuracy:")
    for k, score in overall["topk_scores"].items():
        print(f"  Top-{k}: {score * 100:.2f}%")
    print("\nLegality (greedy argmax):")
    print(f"  Legal moves: {legality_metrics['legality_rate'] * 100:.2f}%")
    print(f"  ({legality_metrics['legal_moves']:,} / {legality_metrics['total_moves']:,})")
    print("=" * 50)


if __name__ == "__main__":
    main()
