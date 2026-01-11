#!/usr/bin/env python3
"""
Upload the chess-transform model to Hugging Face Hub.

This script uploads the model weights (safetensors) and config to a Hugging Face
repository, making it accessible via `load_model_from_hf()`.

Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login to HF: huggingface-cli login
    3. Have model.safetensors and config.json ready

Usage:
    python tools/upload_to_hf.py --repo-id YOUR_USERNAME/chess-gamba-rossa

    # Or with default repo:
    python tools/upload_to_hf.py
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Upload chess model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        default="hadiji/chess-gamba-rossa",
        help="Hugging Face repository ID (e.g., username/repo-name)",
    )
    parser.add_argument(
        "--model-name",
        default="gamba_rossa",
        help="Name of the model to upload (matches config and checkpoint folder)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repository if it doesn't exist",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    config_path = os.path.join(repo_root, f"configs/{args.model_name}.json")
    safetensors_path = os.path.join(
        repo_root, f"checkpoints/{args.model_name}/model.safetensors"
    )
    encoder_path = os.path.join(repo_root, "data/id_to_token.json")

    # Verify files exist
    missing = []
    if not os.path.exists(config_path):
        missing.append(f"Config: {config_path}")
    if not os.path.exists(safetensors_path):
        missing.append(f"Model: {safetensors_path}")
    if not os.path.exists(encoder_path):
        missing.append(f"Encoder: {encoder_path}")

    if missing:
        print("Error: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        print("\nMake sure to:")
        print("  1. Have the model config in configs/")
        print("  2. Convert checkpoint to safetensors:")
        print(
            "     python tools/convert_checkpoint_to_safetensors.py --input checkpoints/gamba_rossa/final.pth"
        )
        print("  3. Have the encoder vocabulary in data/id_to_token.json")
        sys.exit(1)

    api = HfApi()

    # Create repo if requested
    if args.create_repo:
        print(f"Creating repository: {args.repo_id}")
        try:
            create_repo(
                repo_id=args.repo_id,
                private=args.private,
                exist_ok=True,
            )
            print(f"Repository created/verified: https://huggingface.co/{args.repo_id}")
        except Exception as e:
            print(f"Warning: Could not create repo: {e}")

    # Upload files
    print(f"\nUploading to {args.repo_id}...")

    # Upload config.json
    print(f"  Uploading config.json from {config_path}")
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="config.json",
        repo_id=args.repo_id,
    )

    # Upload model.safetensors
    print(f"  Uploading model.safetensors from {safetensors_path}")
    api.upload_file(
        path_or_fileobj=safetensors_path,
        path_in_repo="model.safetensors",
        repo_id=args.repo_id,
    )

    # Upload encoder vocabulary
    print(f"  Uploading id_to_token.json from {encoder_path}")
    api.upload_file(
        path_or_fileobj=encoder_path,
        path_in_repo="id_to_token.json",
        repo_id=args.repo_id,
    )

    # Create and upload README
    readme_content = f"""---
license: mit
tags:
- chess
- transformer
- game-ai
---

# Chess Transformer: {args.model_name}

A ~450M parameter decoder-only Transformer trained to play chess through imitation learning on grandmaster games.

## Quick Start

```python
from chess_seq import load_model_from_hf, MoveEncoder, ChessGameEngine
import torch

# Load model from Hugging Face
model, config = load_model_from_hf("{args.repo_id}")
model.eval()

# Setup encoder and game engine  
encoder = MoveEncoder()
encoder.build()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

engine = ChessGameEngine(model, encoder, device=device)

# Play a game
game, pgn, bad_plies = engine.play_game(n_plies=80, mask_illegal=True)
print(pgn)
```

## Model Details

| Component | Value |
|-----------|-------|
| Parameters | ~450M |
| Architecture | Decoder-only Transformer |
| Layers | 28 |
| Hidden Dim | 1024 |
| Attention Heads | 16 |
| Vocabulary | 4,611 tokens |

## Links

- [GitHub Repository](https://github.com/hadiji/chess-transform)
- [Play on Lichess](https://lichess.org/@/GambaRossa)
"""

    readme_path = os.path.join(repo_root, ".hf_readme_temp.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print("  Uploading README.md")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )
    os.remove(readme_path)

    print(f"\n✅ Upload complete!")
    print(f"   View your model: https://huggingface.co/{args.repo_id}")
    print(f"\n   Users can now load with:")
    print(f'   model, config = load_model_from_hf("{args.repo_id}")')


if __name__ == "__main__":
    main()
