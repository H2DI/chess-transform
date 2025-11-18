"""High-level export shim for the TRM puzzle package.

This module re-exports the main components split into smaller modules:
- `ChessTRMDataset` in `dataset.py`
- building blocks in `blocks.py`
- `TinyRecursiveChessModel` in `core.py`
- training utilities and `main()` in `training.py`

Keep imports external code stable by importing here.
"""

from .dataset import ChessTRMDataset
from .blocks import RMSNorm, SwiGLU, TinyMLPBlock, ReasoningNet
from .core import TinyRecursiveChessModel
from .training import train_one_epoch, evaluate, main

__all__ = [
    "ChessTRMDataset",
    "RMSNorm",
    "SwiGLU",
    "TinyMLPBlock",
    "ReasoningNet",
    "TinyRecursiveChessModel",
    "train_one_epoch",
    "evaluate",
    "main",
]
