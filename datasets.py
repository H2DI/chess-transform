import torch
from torch.utils.data import Dataset, IterableDataset
import csv
import numpy as np

import chess
import chess.pgn


"""
TODO: 
sort games by  length
filter games that end in checkmate
"""


def parse_pgn_game(game, encoder):
    """
    alternatives:
    - add piece info
    - encode jointly to and from
    """
    board = game.board()
    moves_list = ["START"]
    for move in game.mainline_moves():
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        # print(from_square, to_square)
        moves_list += [from_square, to_square]
        board.push(move)
    moves_list.append("END")
    return encoder.transform(moves_list)


def process_pgn_file(input_file, output_file, encoder, T=10, start_id=0):
    with open(input_file, "r") as pgn, open(output_file, "w", newline="") as out:
        csv_writer = csv.writer(out)
        csv_writer.writerow(["game_id", "moves"])

        game_id = start_id
        while True and game_id < T:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            encoded_moves = parse_pgn_game(game, encoder)
            csv_writer.writerow(
                [game_id, " ".join([str(move) for move in encoded_moves])]
            )
            game_id += 1


class ChessDataset(IterableDataset):
    def __init__(self, csv_file):
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
                full_line = self.parse_line(line)
                for i in range(1, len(full_line)):
                    yield full_line[:i]
