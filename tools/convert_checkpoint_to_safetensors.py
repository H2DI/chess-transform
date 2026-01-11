#!/usr/bin/env python3
"""
Minimal converter: assumes the checkpoint is a dict containing a
`model_state_dict` key and writes a `.safetensors` file with the tensors.

Example:
  python scripts/convert_checkpoint_to_safetensors.py \
      --input checkpoints/gamba_rossa/final.pth

The output will be `checkpoints/gamba_rossa/final.safetensors` by default.
"""

import argparse
import os
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="checkpoints/gamba_rossa/final.pth")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    # allow older checkpoints that were pickled with top-level module name
    # `configs` by mapping it to the package configs module before loading
    import sys

    try:
        import chess_seq.configs as _cs_configs

        sys.modules.setdefault("configs", _cs_configs)
    except Exception:
        # if chess_seq isn't installed in the environment, proceed without shim
        pass

    # load with weights_only=False to allow older checkpoint formats that
    # reference project-local classes (trusted source required)
    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError("Expected a checkpoint dict containing 'model_state_dict'")

    state_dict = ckpt["model_state_dict"]

    # Strip _orig_mod. prefix if present (from torch.compile)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        print("Stripped _orig_mod. prefix from keys")

    # Clone tensors to avoid shared-storage issues when writing safetensors
    tensors = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            tensors[k] = v.detach().cpu().clone()
        elif isinstance(v, torch.Tensor):
            tensors[k] = v.cpu().clone()
        else:
            # skip non-tensor values
            continue

    out = args.output or os.path.splitext(args.input)[0] + ".safetensors"
    try:
        save_file(tensors, out)
        print(f"Saved safetensors: {out}")
    except RuntimeError as e:
        msg = str(e)
        if "share memory" in msg or "share" in msg:
            # fallback to save_model which can handle shared storage
            try:
                from safetensors.torch import save_model

                save_model(tensors, out)
                print(f"Saved safetensors with save_model fallback: {out}")
            except Exception as e2:
                raise RuntimeError(f"Failed to save with fallback: {e2}")
        else:
            raise


if __name__ == "__main__":
    main()
