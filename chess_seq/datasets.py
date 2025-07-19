import torch
from torch.utils.data import IterableDataset


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
