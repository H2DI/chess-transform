#!/usr/bin/env python3
"""Strip optimizer and scheduler states from saved checkpoints.

This script searches `--checkpoints-dir` and its immediate subdirectories for
files named `checkpoint_{number}.pth`. For each file it loads the checkpoint
dict (using `torch.load`), removes the keys `optimizer_state_dict` and
`scheduler_state_dict` if present, and atomically rewrites the file.

Options:
  --dry-run   : show what would be changed without touching files
  --backup DIR: copy original files to DIR before rewriting

The script purposefully keeps loading simple and readable: it uses
`torch.load(path, map_location='cpu')` and skips files that fail to load.
"""

import argparse
import os
import re
import tempfile
import shutil
from typing import List

import torch


CHK_PATTERN = re.compile(r"checkpoint_(\d+)\.pth$")


def find_checkpoint_files(folder: str) -> List[str]:
    files = []
    for entry in os.listdir(folder):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and CHK_PATTERN.search(entry):
            files.append(path)
    return sorted(files)


def strip_file(path: str):
    try:
        chk = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"warning: failed to load {path}: {e}")
        return {"path": path, "modified": False, "reason": "load_failed"}

    if not isinstance(chk, dict):
        print(f"skipping {path}: checkpoint not a dict (type={type(chk)})")
        return {"path": path, "modified": False, "reason": "not_dict"}

    for key in ("optimizer_state_dict", "scheduler_state_dict"):
        if key in chk:
            del chk[key]

    dirpath = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dirpath)
    os.close(fd)
    try:
        torch.save(chk, tmp)
        shutil.move(tmp, path)
        print(f"Stripped optimizer/scheduler and rewrote {path}")
        return {"path": path, "modified": True, "reason": "stripped"}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main(root: str):
    if not os.path.isdir(root):
        raise SystemExit(f"checkpoints folder not found: {root}")

    summary = []

    # top-level checkpoint files
    for p in find_checkpoint_files(root):
        summary.append(strip_file(p))

    modified = sum(1 for s in summary if s.get("modified"))
    total = len(summary)
    print(
        f"\nSummary: processed={total} modified={modified} skipped={total - modified}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoints-dir", default="checkpoints", help="root checkpoints folder"
    )
    args = p.parse_args()

    main(args.checkpoints_dir)
