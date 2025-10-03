import torch
from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np

"""
TODO: 
sort games by  length
filter games that end in checkmate
"""


class ChessCSVDataset(IterableDataset):
    def __init__(self, csv_file, device=None):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.csv_file = csv_file

    def parse_line(self, line):
        game_id, moves = line.strip().split(",", 1)
        move_list = moves.split(" ")
        moves_tensor = torch.tensor([int(x) for x in move_list], requires_grad=False)
        return moves_tensor

    def __iter__(self):
        with open(self.csv_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                yield self.parse_line(line)


class ChessNPZDataset(IterableDataset):
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
    train_file,
    batch_size=16,
    device=None,
    padding_value=-1,
    max_length=1200,
    data_format="npz",
):
    """Build a DataLoader for the ChessDataset."""
    if data_format == "npz":
        dataset = ChessNPZDataset(train_file, device=device)
    elif data_format == "csv":
        dataset = ChessCSVDataset(train_file, device=device)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")

    def collate_fn(batch_list):
        batch_list = [seq[:max_length] for seq in batch_list]
        inputs_padded = pad_sequence(
            batch_list, batch_first=True, padding_value=padding_value
        )
        return inputs_padded

    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=False
    )
