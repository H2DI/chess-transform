#!/usr/bin/env python3
"""Prune checkpoint files under `checkpoints/`.

Behaviour:
- For each checkpoint file matching `checkpoint_*.pth` in each subdirectory of the
  provided `--checkpoints-dir`, keep one every `--keep-every` files (sorted by
  numeric suffix). All other checkpoint files are deleted.
- For kept checkpoint files, the script will remove the keys
  `optimizer_state_dict` and `scheduler_state_dict` from the saved dict (if present)
  and overwrite the checkpoint file. This reduces disk usage and speeds up
  later loading for inference.

This script is destructive by default. Use `--dry-run` to preview actions.
"""

import argparse
import os
import re
import tempfile
import shutil
from typing import List

import torch


CHK_PATTERN = re.compile(r"checkpoint_(\d+)\.pth$")


def numeric_key(filename: str):
    m = CHK_PATTERN.search(filename)
    if not m:
        return None
    return int(m.group(1))


def find_checkpoint_files(folder: str) -> List[str]:
    files = []
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and CHK_PATTERN.search(entry):
            files.append(path)
    return files


def prune_dir(dirpath: str, keep_every: int = 4, dry_run: bool = False):
    files = find_checkpoint_files(dirpath)
    if not files:
        return {
            "dir": dirpath,
            "kept": 0,
            "deleted": 0,
        }

    # sort by numeric suffix
    files_sorted = sorted(files, key=lambda p: numeric_key(os.path.basename(p)) or 0)

    kept = 0
    deleted = 0
    for idx, path in enumerate(files_sorted):
        keep = idx % keep_every == 0
        if keep:
            kept += 1
            if dry_run:
                print(f"[DRY] would KEEP {path}")
            else:
                # load and strip optimizer/scheduler states
                try:
                    chk = torch.load(path, map_location="cpu")
                except Exception as e:
                    print(f"warning: failed to load {path}: {e}")
                    continue

                removed_any = False
                for key in ["optimizer_state_dict", "scheduler_state_dict"]:
                    if key in chk:
                        del chk[key]
                        removed_any = True

                if removed_any:
                    # write atomically
                    fd, tmp = tempfile.mkstemp(dir=dirpath)
                    os.close(fd)
                    try:
                        torch.save(chk, tmp)
                        shutil.move(tmp, path)
                        print(f"Stripped optimizer/scheduler and rewrote {path}")
                    finally:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                else:
                    print(f"Kept (no optimizer/scheduler present): {path}")
        else:
            deleted += 1
            if dry_run:
                print(f"[DRY] would DELETE {path}")
            else:
                try:
                    os.remove(path)
                    print(f"Deleted {path}")
                except Exception as e:
                    print(f"warning: failed to delete {path}: {e}")

    return {"dir": dirpath, "kept": kept, "deleted": deleted}


def main(root_checkpoints: str, keep_every: int = 4, dry_run: bool = False):
    if not os.path.isdir(root_checkpoints):
        raise SystemExit(f"checkpoints folder not found: {root_checkpoints}")

    # process direct checkpoint files in root and then subdirectories
    summaries = []

    # root-level
    summaries.append(
        prune_dir(root_checkpoints, keep_every=keep_every, dry_run=dry_run)
    )

    # subdirectories
    for name in sorted(os.listdir(root_checkpoints)):
        sub = os.path.join(root_checkpoints, name)
        if os.path.isdir(sub):
            summaries.append(prune_dir(sub, keep_every=keep_every, dry_run=dry_run))

    print("\nSummary:")
    total_kept = sum(s.get("kept", 0) for s in summaries)
    total_deleted = sum(s.get("deleted", 0) for s in summaries)
    for s in summaries:
        print(f"{s['dir']}: kept={s['kept']} deleted={s['deleted']}")
    print(f"TOTAL kept={total_kept} deleted={total_deleted}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints-dir", default="checkpoints", help="root checkpoints folder"
    )
    p.add_argument(
        "--keep-every",
        type=int,
        default=4,
        help="Keep one every N checkpoints (by sorted order)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without changing files",
    )
    args = p.parse_args()

    main(args.checkpoints_dir, keep_every=args.keep_every, dry_run=args.dry_run)
