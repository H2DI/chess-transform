"""
Analyze probe performance across all layers.
Trains probes on multiple layers and generates a comparison report.

Usage:
    python analyze_layers.py --probe-type linear --layers 0 7 14 21 27
    python analyze_layers.py --probe-type linear --all-layers
"""

import argparse
import os
import torch
import json

from chess_seq.models import ChessNet
from chess_seq.encoder import MoveEncoder
from chess_seq.evaluation.probes.probes import create_probe
from chess_seq.evaluation.probes.probe_training import ProbeDataset, train_probe
from chess_seq.utils import save_and_load


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze probes across layers")

    parser.add_argument(
        "--model-name",
        type=str,
        default="gamba_gialla",
        help="Name of the trained model",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Type of probe",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to analyze (e.g., 0 7 14 21 27)",
    )
    parser.add_argument("--all-layers", action="store_true", help="Analyze all layers")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train_npz/shard_00000.npz",
        help="Training data",
    )
    parser.add_argument("--val-data", type=str, default=None, help="Validation data")
    parser.add_argument(
        "--max-samples", type=int, default=20000, help="Max samples per layer"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per probe")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/layer_analysis",
        help="Output directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    checkpoint_path = save_and_load.get_latest_checkpoint(args.model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_config = checkpoint["model_config"]
    model = ChessNet(config=model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(f"Loaded model: {model_config.name}")
    print(f"Layers: {model_config.n_layers}")

    # Load encoder
    encoder = MoveEncoder()
    encoder.load(model_config.encoder_path)

    # Determine layers to analyze
    if args.all_layers:
        layers = list(range(model_config.n_layers))
    elif args.layers:
        layers = args.layers
    else:
        # Default: sample across depth
        num_samples = 5
        layers = [i * model_config.n_layers // num_samples for i in range(num_samples)]

    print(f"\nAnalyzing layers: {layers}")

    # Results storage
    results = {
        "model_name": args.model_name,
        "probe_type": args.probe_type,
        "layers": layers,
        "layer_results": {},
    }

    # Train probe for each layer
    print("\n" + "=" * 60)
    print("TRAINING PROBES")
    print("=" * 60)

    for layer_idx in layers:
        print(f"\n{'=' * 60}")
        print(f"LAYER {layer_idx}/{model_config.n_layers - 1}")
        print(f"{'=' * 60}")

        # Create dataset
        print("Creating dataset...")
        dataset = ProbeDataset(
            args.train_data,
            model,
            encoder,
            layer_idx=layer_idx,
            max_samples=args.max_samples,
            device=args.device,
            cache_activations=True,
        )

        val_dataset = None
        if args.val_data:
            val_dataset = ProbeDataset(
                args.val_data,
                model,
                encoder,
                layer_idx=layer_idx,
                max_samples=args.max_samples // 4,
                device=args.device,
                cache_activations=True,
            )

        # Create probe
        if args.probe_type == "mlp":
            probe = create_probe("mlp", model_config.k, hidden_dim=512)
        else:
            probe = create_probe("linear", model_config.k)

        # Train
        save_path = os.path.join(
            args.output_dir, f"{args.model_name}_{args.probe_type}_layer{layer_idx}.pth"
        )

        history = train_probe(
            probe,
            dataset,
            val_dataset,
            num_epochs=args.epochs,
            batch_size=128,
            lr=1e-3,
            device=args.device,
            save_path=save_path,
        )

        # Store results
        results["layer_results"][layer_idx] = {
            "train_square_acc": history["train_square_acc"][-1],
            "train_board_acc": history["train_board_acc"][-1],
            "val_square_acc": history["val_square_acc"][-1] if val_dataset else None,
            "val_board_acc": history["val_board_acc"][-1] if val_dataset else None,
            "history": history,
        }

        print(f"\nLayer {layer_idx} Results:")
        print(f"  Train Square Acc: {history['train_square_acc'][-1]:.4f}")
        print(f"  Train Board Acc: {history['train_board_acc'][-1]:.4f}")
        if val_dataset:
            print(f"  Val Square Acc: {history['val_square_acc'][-1]:.4f}")
            print(f"  Val Board Acc: {history['val_board_acc'][-1]:.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, "layer_analysis_results.json")
    with open(results_path, "w") as f:
        # Convert history to serializable format
        serializable_results = results.copy()
        for layer_idx in serializable_results["layer_results"]:
            layer_data = serializable_results["layer_results"][layer_idx]
            # Keep only final metrics, not full history
            layer_data.pop("history", None)
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nSquare Accuracy by Layer:")
    print(f"{'Layer':<10} {'Train Acc':<15} {'Val Acc':<15}")
    print("-" * 40)
    for layer_idx in sorted(results["layer_results"].keys()):
        r = results["layer_results"][layer_idx]
        val_acc_str = f"{r['val_square_acc']:.4f}" if r["val_square_acc"] else "N/A"
        print(f"{layer_idx:<10} {r['train_square_acc']:<15.4f} {val_acc_str:<15}")

    # Find best layer
    if val_dataset:
        best_layer = max(
            results["layer_results"].keys(),
            key=lambda k: results["layer_results"][k]["val_square_acc"],
        )
        best_acc = results["layer_results"][best_layer]["val_square_acc"]
    else:
        best_layer = max(
            results["layer_results"].keys(),
            key=lambda k: results["layer_results"][k]["train_square_acc"],
        )
        best_acc = results["layer_results"][best_layer]["train_square_acc"]

    print(f"\nBest Layer: {best_layer} (accuracy: {best_acc:.4f})")

    # Generate plot
    try:
        import matplotlib.pyplot as plt

        layer_indices = sorted(results["layer_results"].keys())
        train_accs = [
            results["layer_results"][i]["train_square_acc"] for i in layer_indices
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(
            layer_indices, train_accs, "o-", linewidth=2, markersize=8, label="Train"
        )

        if val_dataset:
            val_accs = [
                results["layer_results"][i]["val_square_acc"] for i in layer_indices
            ]
            plt.plot(
                layer_indices, val_accs, "s-", linewidth=2, markersize=8, label="Val"
            )

        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Square Accuracy", fontsize=12)
        plt.title(f"Probe Performance Across Layers ({args.probe_type})", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        plot_path = os.path.join(args.output_dir, "layer_analysis_plot.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")

    except ImportError:
        print("\nmatplotlib not available, skipping plot")

    print(f"\nResults saved to: {results_path}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
