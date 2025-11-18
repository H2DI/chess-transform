# trm_puzzle

Small package containing the Tiny Recursive Model (TRM) for chess puzzles.

This directory contains a compact dataset loader, a tiny TRM model implementation,
training utilities, and helper building blocks. The package layout was split from
the original single-file `model.py` into smaller modules for clarity.

Quick overview
- `trm_puzzle/dataset.py`  - `ChessTRMDataset` loader for `.npz` puzzle datasets
- `trm_puzzle/blocks.py`   - small neural building blocks (RMSNorm, SwiGLU, MLP)
- `trm_puzzle/core.py`     - `TinyRecursiveChessModel` core model
- `trm_puzzle/training.py` - training utilities and `main()` training loop
- `trm_puzzle/model.py`    - a lightweight shim re-exporting public symbols
- `trm_puzzle/config.py`   - default configuration (data path, model and optim)

Requirements

- Python 3.8+
- PyTorch (version appropriate for your CUDA / CPU setup)
- NumPy

Install (pip)

```bash
python3 -m pip install --user torch numpy
```

If you use conda, create/activate an environment and install torch via the
recommended conda channel for your platform.

Running training

The training script reads configuration from `trm_puzzle/config.py` by default.
By default it expects a dataset at `trm_puzzle/data/mate_in_1.npz` (set in
`Config.data.file`).

Run from the repository root:

```bash
# using the package entrypoint
python -m trm_puzzle.training

# or run the module script directly
python trm_puzzle/training.py
```

The training module currently uses the default config values. For quick
experimentation you can create a small subset `.npz` and point the config to it
or run an interactive test (example below).

Creating a small test dataset

If you don't want to run on the full dataset, you can create a tiny `.npz` to
test the code. Example Python snippet (run from repo root):

```python
import numpy as np

# load the full npz (if you have it)
data = np.load('trm_puzzle/data/mate_in_1.npz', allow_pickle=True)

# take first 1024 examples (or create synthetic ones)
idx = np.arange(min(1024, len(data['X'])))
small = {k: data[k][idx] for k in data.files}
np.savez_compressed('trm_puzzle/data/mate_in_1_small.npz', **small)

# update trm_puzzle/config.py -> Config.data.file to point to the small file
```

Importing the components

You can import the main classes from the package shim:

```python
from trm_puzzle import ChessTRMDataset, TinyRecursiveChessModel, train_one_epoch

# or import directly from modules
from trm_puzzle.dataset import ChessTRMDataset
from trm_puzzle.core import TinyRecursiveChessModel
```

Notes and next improvements

- Add CLI flags to `trm_puzzle/training.py` (data path, epochs, batch size,
  debug/test mode).
- Add unit tests and a small fixture dataset for CI/quick smoke tests.
- Add clearer logging and checkpoint saving for training.

If you'd like, I can add a `--limit`/`--dry-run` CLI to `training.py` next.
