import torch
from torch.utils.data import IterableDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

"""
TODO: 
sort games by  length
filter games that end in checkmate
"""


class ChessDataset(IterableDataset):
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


def build_dataloader(csv_train, batch_size=16, device=None, padding_value=-1):
    """Build a DataLoader for the ChessDataset."""
    dataset = ChessDataset(csv_train, device=device)

    def collate_fn(batch_list):
        inputs_padded = pad_sequence(
            batch_list, batch_first=True, padding_value=padding_value
        )
        return inputs_padded

    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=False
    )
