"""
Train a probe to recover chess board positions from model internal states.

This script trains a linear or MLP probe to decode the current board position
from the activations of a trained chess transformer model.

Usage:
    python train_probe.py --layer 14 --probe-type linear
    python train_probe.py --layer 27 --probe-type mlp --hidden-dim 512
    python train_probe.py --all-layers --probe-type layerwise
"""

import argparse
import os
import torch

from chess_seq import ChessNet, MoveEncoder, save_and_load
from chess_seq.evaluation.probes.probes import create_probe
from chess_seq.evaluation.probes.probe_training import (
    ProbeDataset,
    train_probe,
    evaluate_probe,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a probe to decode board positions"
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="gamba_rossa",
        help="Name of the trained model to probe",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final.pth",
        help="Specific checkpoint path (default: use latest)",
    )

    # Probe configuration
    parser.add_argument(
        "--probe-type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "sequence", "layerwise"],
        help="Type of probe to train",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Which layer to extract activations from (default: middle layer)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Use activations from all layers (for layerwise probe)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for MLP/sequence probes",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="attention",
        choices=["attention", "mean", "max", "last"],
        help="Aggregation method for sequence probe",
    )

    # Data configuration
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train_npz/shard_00000.npz",
        help="Training data file",
    )
    parser.add_argument(
        "--val-data", type=str, default=None, help="Validation data file (optional)"
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50000,
        help="Maximum training samples to generate",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=5000,
        help="Maximum validation samples to generate",
    )
    parser.add_argument(
        "--no-cache-activations",
        action="store_false",
        dest="cache_activations",
        help="Don't cache activations in memory (slower but uses less RAM)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/probes",
        help="Directory to save probe checkpoints",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate a trained probe (requires --checkpoint)",
    )

    return parser.parse_args()


def load_model_for_probe(
    model_name: str, checkpoint_path: str = None, device: str = "cpu"
):
    """Load a trained ChessNet model."""
    # Use the existing utils.load_model function
    if checkpoint_path is None:
        # Load latest checkpoint or final model
        try:
            model, model_config, info = save_and_load.load_model_from_checkpoint(
                model_name, special_name="final"
            )
        except FileNotFoundError:
            # If 'final' doesn't exist, try loading the latest checkpoint
            model, model_config, info = save_and_load.load_model_from_checkpoint(
                model_name
            )
    else:
        # Load specific checkpoint by parsing the path
        filename = os.path.basename(checkpoint_path).replace(".pth", "")
        if filename.startswith("checkpoint_"):
            number = int(filename.split("_")[1])
            model, model_config, info = save_and_load.load_model_from_checkpoint(
                model_name, number=number
            )
        else:
            model, model_config, info = save_and_load.load_model_from_checkpoint(
                model_name, special_name=filename
            )

    model = model.to(device)
    model.eval()

    print(f"Loaded model: {model_config.name}")
    print(f"  Layers: {model_config.n_layers}")
    print(f"  Hidden dim: {model_config.k}")
    print(f"  Heads: {model_config.n_head}")

    return model, model_config


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and encoder
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    model, model_config = load_model_for_probe(
        args.model_name, args.checkpoint, args.device
    )

    encoder = MoveEncoder()
    encoder.load(model_config.encoder_path)

    # Determine layer(s) to probe
    if args.all_layers:
        layer_indices = list(range(model_config.n_layers))
        layer_str = "all"
    else:
        layer_indices = [args.layer]
        layer_str = f"layer{args.layer}"

    # Create probe
    print("\n" + "=" * 60)
    print("CREATING PROBE")
    print("=" * 60)

    if args.probe_type == "layerwise":
        probe = create_probe(
            args.probe_type,
            input_dim=model_config.k,
            num_layers=len(layer_indices),
            hidden_dim=args.hidden_dim,
        )
        print(f"Created layerwise probe with {len(layer_indices)} layers")
    elif args.probe_type == "sequence":
        probe = create_probe(
            args.probe_type,
            input_dim=model_config.k,
            hidden_dim=args.hidden_dim,
            aggregation=args.aggregation,
        )
        print(f"Created sequence probe with {args.aggregation} aggregation")
    elif args.probe_type == "mlp":
        probe = create_probe(
            args.probe_type, input_dim=model_config.k, hidden_dim=args.hidden_dim
        )
        print(f"Created MLP probe with hidden_dim={args.hidden_dim}")
    else:  # linear
        probe = create_probe(args.probe_type, input_dim=model_config.k)
        print("Created linear probe")

    num_params = sum(p.numel() for p in probe.parameters())
    print(f"Probe parameters: {num_params:,}")

    # Probe checkpoint path
    probe_name = f"{args.model_name}_{args.probe_type}_{layer_str}"
    probe_path = os.path.join(args.output_dir, f"{probe_name}.pth")

    if args.evaluate_only:
        # Load trained probe and evaluate
        print("\n" + "=" * 60)
        print("LOADING TRAINED PROBE")
        print("=" * 60)
        checkpoint = torch.load(probe_path, map_location="cpu")
        probe.load_state_dict(checkpoint["probe_state_dict"])
        print(f"Loaded probe from: {probe_path}")
        print(f"Training square accuracy: {checkpoint['square_accuracy']:.4f}")

    # Create datasets
    print("\n" + "=" * 60)
    print("CREATING DATASETS")
    print("=" * 60)

    if args.probe_type == "layerwise":
        # TODO: Implement dataset for layerwise probes
        print("Note: Layerwise probes not fully implemented in dataset yet")
        print("Using single layer for now...")
        layer_idx = layer_indices[len(layer_indices) // 2]  # Use middle layer
    else:
        layer_idx = layer_indices[0]

    train_dataset = ProbeDataset(
        args.train_data,
        model,
        encoder,
        layer_idx=layer_idx,
        max_samples=args.max_train_samples,
        device=args.device,
        cache_activations=args.cache_activations,
    )

    val_dataset = None
    if args.val_data:
        val_dataset = ProbeDataset(
            args.val_data,
            model,
            encoder,
            layer_idx=layer_idx,
            max_samples=args.max_val_samples,
            device=args.device,
            cache_activations=args.cache_activations,
        )

    if not args.evaluate_only:
        # Train probe
        print("\n" + "=" * 60)
        print("TRAINING PROBE")
        print("=" * 60)

        history = train_probe(
            probe,
            train_dataset,
            val_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_path=probe_path,
        )

        # Save final model and history
        final_path = os.path.join(args.output_dir, f"{probe_name}_final.pth")
        torch.save(
            {
                "probe_state_dict": probe.state_dict(),
                "history": history,
                "args": vars(args),
                "model_config": model_config,
            },
            final_path,
        )
        print(f"\nSaved final probe to: {final_path}")

        # Plot training curves if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(history["train_loss"], label="Train")
            if val_dataset:
                axes[0].plot(history["val_loss"], label="Val")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training Loss")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(history["train_square_acc"], label="Train")
            if val_dataset:
                axes[1].plot(history["val_square_acc"], label="Val")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Square Accuracy")
            axes[1].set_title("Square-Level Accuracy")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            plot_path = os.path.join(args.output_dir, f"{probe_name}_training.png")
            plt.savefig(plot_path)
            print(f"Saved training plot to: {plot_path}")
        except ImportError:
            print("matplotlib not available, skipping plot")

    # Evaluate probe
    print("\n" + "=" * 60)
    print("EVALUATING PROBE")
    print("=" * 60)

    evaluate_probe(
        probe,
        train_dataset if args.evaluate_only else (val_dataset or train_dataset),
        device=args.device,
        num_samples_to_visualize=3,
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
