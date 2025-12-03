#!/usr/bin/env python3
"""Report totals for NPZ shards in `data/train_npz`.

Outputs:
- total number of games across all shards in the folder
- number of token entries (sum of moves) in the first shard (default `shard_00000.npz`)

Usage:
  python3 scripts/report_train_npz.py --dir data/train_npz
"""

import os
import argparse
import glob
import numpy as np


def main(folder, first_shard_name="shard_00000.npz"):
    if not os.path.isdir(folder):
        print(f"error: folder not found: {folder}")
        raise SystemExit(2)

    files = sorted(
        [
            p
            for p in glob.glob(os.path.join(folder, "*.npz"))
            if os.path.basename(p) != ".DS_Store"
        ]
    )
    if not files:
        print(f"no .npz files found in {folder}")
        return

    total_games = 0
    total_tokens_all_shards = 0
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
        except Exception as e:
            print(f"warning: failed to load {f}: {e}")
            continue

        if "game_ids" in data:
            total_games += int(len(data["game_ids"]))

        if "tokens" in data:
            # if game_ids not present, use tokens length for games
            if "game_ids" not in data:
                total_games += int(len(data["tokens"]))

            # also count tokens across this shard
            try:
                for seq in data["tokens"]:
                    total_tokens_all_shards += int(len(seq))
            except Exception:
                pass

    first_shard_path = os.path.join(folder, first_shard_name)
    if not os.path.exists(first_shard_path):
        # fallback to first file in sorted listing
        first_shard_path = files[0]

    data = np.load(first_shard_path, allow_pickle=True)
    tokens = data.get("tokens", None)

    if tokens is None:
        print(f"first shard {first_shard_path} has no 'tokens' array")
        tokens_count = 0
    else:
        # tokens is an object array of sequences; sum their lengths
        tokens_count = 0
        for seq in tokens:
            try:
                tokens_count += int(len(seq))
            except Exception:
                # if stored as scalar or invalid, ignore
                pass

    print(f"total_games_across_shards: {total_games}")
    print(f"first_shard: {os.path.basename(first_shard_path)}")
    print(f"tokens_in_first_shard: {tokens_count}")
    print(f"total_tokens_across_shards: {total_tokens_all_shards}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="data/train_npz",
        help="folder containing shard_*.npz files",
    )
    parser.add_argument(
        "--first",
        type=str,
        default="shard_00000.npz",
        help="filename of first shard to inspect (relative to --dir)",
    )
    args = parser.parse_args()

    main(args.dir, first_shard_name=args.first)
