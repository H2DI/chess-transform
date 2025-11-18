import numpy as np
import torch
from torch.utils.data import Dataset


class ChessTRMDataset(Dataset):
    """
    Expects npz with:
      - ids: (N,) array of puzzle ids (strings)
      - X:   (N, 69) int16 position tokens
      - Y:   (N, 3)  int16 [from, to, promo]
    Builds:
      - move_to_id: {(from, to, promo) -> label_id}
      - labels: (N,) int64 of label indices in [0, num_moves)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.ids = data["ids"]
        self.X = data["X"].astype(np.int64)  # 69 ints
        self.Y = data["Y"].astype(np.int64)  # 1 int

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])  # (69,)
        y = torch.tensor(self.Y[idx])  # scalar label
        return x, y
