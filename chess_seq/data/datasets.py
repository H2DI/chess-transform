import torch
from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import os

"""
TODO: 
sort games by  length
filter games that end in checkmate
"""


class ChessDataset(IterableDataset):
    def __init__(self, npz_path, device=None, shuffle=False):
        """
        npz_path: path to dataset_XXXXX.npz file
        shuffle: randomize order of samples
        """
        self.npz_path = npz_path
        self.shuffle = shuffle

        if device is None:
            device = torch.device("cpu")
        self.device = device

        # Load entire file once (tokens is object array)
        data = np.load(npz_path, allow_pickle=True)

        self.game_ids = data["game_ids"]  # (N,)
        self.tokens = data["tokens"]  # (N,) object array
        self.num_samples = len(self.tokens)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in indices:
            yield torch.tensor(self.tokens[i], device=self.device)


def build_dataloader(
    npz_path,
    batch_size=16,
    device=None,
    padding_value=-1,
    max_length=500,
    shuffle=False,
):
    dataset = ChessDataset(
        npz_path=npz_path,
        device=device,
        shuffle=shuffle,
    )

    def collate_fn(batch_list):
        # truncate
        batch_list = [seq[:max_length] for seq in batch_list]

        # pad
        padded = pad_sequence(batch_list, batch_first=True, padding_value=padding_value)
        return padded

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
    )
