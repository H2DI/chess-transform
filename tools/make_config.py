import os
import sys
from chess_seq import ModelConfig

if __name__ == "__main__":
    model_name = "gamba_rossa"

    cfg = ModelConfig(
        name=model_name,
        vocab_size=4611,
        block_size=256,
        k=1024,
        head_dim=128,
        n_head=16,
        n_layers=28,
        dropout=0.0,
        kv_groups=2,
        ff_expansion=3,
        pad_index=4610,
        special_freqs=[2],
        encoder_path=f"checkpoints/{model_name}/id_to_token.json",
    )

    save_path = f"checkpoints/{model_name}/{model_name}.json"
    if os.path.exists(save_path):
        resp = (
            input(f"'{save_path}' already exists. Overwrite? [y/N]: ").strip().lower()
        )
        if resp not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    cfg.save()
