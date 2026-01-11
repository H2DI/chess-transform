"""
Evaluate a saved probe checkpoint (created by train_probe.py).

Usage:
    python evaluate_saved_probe.py --probe checkpoints/probes/gamba_rossa_linear_layer14_final.pth

This script will:
- load the probe checkpoint (allowing ModelConfig safe global)
- load the corresponding ChessNet model (final) via utils.load_model
- create a ProbeDataset using the data file specified in the probe args
- run evaluate_probe and print metrics + sample visualizations
"""

import argparse
import torch
import os
import chess_seq.configs as cs
from chess_seq.utils import save_and_load
from chess_seq.encoder import MoveEncoder
from chess_seq.evaluation.probes import create_probe
from chess_seq.evaluation.probe_training import ProbeDataset, evaluate_probe


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", required=True, help="Path to saved probe checkpoint")
    p.add_argument(
        "--data-file", default=None, help="Dataset file to use (overrides checkpoint)"
    )
    p.add_argument(
        "--max-samples", type=int, default=None, help="Max samples for dataset"
    )
    p.add_argument("--device", default="cpu", help="Device to run on")
    return p.parse_args()


def main():
    args = parse_args()
    probe_path = args.probe
    device = args.device

    if not os.path.exists(probe_path):
        raise FileNotFoundError(probe_path)

    # Load probe checkpoint allowing ModelConfig global
    with torch.serialization.safe_globals([cs.ModelConfig]):
        ck = torch.load(probe_path, map_location="cpu", weights_only=False)

    print("Checkpoint keys:", list(ck.keys()))

    # Get probe metadata
    probe_state = ck.get("probe_state_dict")
    probe_args = ck.get("args", {})
    model_config = ck.get("model_config")

    if model_config is None:
        # Try to infer model name from probe filename
        # filename format: <model>_<probe>_layer<ID>_final.pth
        fname = os.path.basename(probe_path)
        model_name = fname.split("_")[0]
    else:
        model_name = model_config.name

    print("Using model:", model_name)

    # Load chess model (final)
    print("\nLoading ChessNet model...")
    try:
        model, mc, info = save_and_load.load_model_from_checkpoint(
            model_name, special_name="final"
        )
    except Exception:
        # fallback: try without special_name
        model, mc, info = save_and_load.load_model_from_checkpoint(model_name)
    model = model.to(device)
    model.eval()

    # Determine probe type and create probe
    probe_type = probe_args.get("probe_type", "linear")
    hidden_dim = probe_args.get("hidden_dim", 512)

    probe = create_probe(
        probe_type,
        input_dim=model.k if hasattr(model, "k") else getattr(mc, "k", None),
        hidden_dim=hidden_dim,
    )
    probe.load_state_dict(probe_state)
    probe.to(device)
    probe.eval()

    # Build encoder
    encoder = MoveEncoder()
    encoder.load(mc.encoder_path)

    # Determine data file
    data_file = (
        args.data_file
        or probe_args.get("train_data")
        or "data/train_npz/shard_00000.npz"
    )

    # Determine layer index
    layer_idx = probe_args.get("layer", 14)

    print(f"\nCreating dataset from {data_file} (layer {layer_idx})...")
    dataset = ProbeDataset(
        data_file,
        model,
        encoder,
        layer_idx=layer_idx,
        max_samples=(args.max_samples or probe_args.get("max_train_samples")),
        device=device,
        cache_activations=probe_args.get("cache_activations", True),
    )

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate_probe(probe, dataset, device=device, num_samples_to_visualize=5)
    print("\nDone. Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
