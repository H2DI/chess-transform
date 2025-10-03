import torch
from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np


class TTTDataset(IterableDataset):
    def __init__(self, npz_file, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.npz_file = npz_file

    def __iter__(self):
        with np.load(self.npz_file) as data:
            for line in data:
                yield torch.tensor(data[line], device=self.device)


def build_dataloader(
    train_file, batch_size=16, device=None, padding_value=-1, max_length=11
):
    """Build a DataLoader for the ChessDataset."""
    dataset = TTTDataset(train_file, device=device)

    def collate_fn(batch_list):
        batch_list = [seq[:max_length] for seq in batch_list]
        inputs_padded = pad_sequence(
            batch_list, batch_first=True, padding_value=padding_value
        )
        return inputs_padded

    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=False
    )
