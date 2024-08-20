import torch
from torch.utils.data import Dataset
import pickle

import chess
import chess.pgn

from itertools import islice

# import time


def consume(iterator, n):
    next(islice(iterator, n, n), None)


def parse_pgn(game, encoder):
    board = game.board()
    moves = [""]
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    return encoder.transform(moves)


def generate_games_list(first_game=0, n_games=10):
    # a = time.time()
    last_game = first_game + n_games
    with open("data/move_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    # print(time.time() - a)
    parsed_games = []
    with open("data/lichess_elite_2024-06.pgn") as pgn:
        consume(pgn, n=first_game)
        for _ in range(first_game, last_game):
            game = chess.pgn.read_game(pgn)
            parsed_games.append(parse_pgn(game, encoder))
    return parsed_games


class NextMoveDataset(Dataset):
    def __init__(self, games):
        self.data = []
        for game in games:
            for i in range(1, len(game)):
                # Each pair consists of (input_sequence, target)
                self.data.append((game[:i], game[i]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_sequence, target = self.data[idx]
        return torch.from_numpy(input_sequence), torch.tensor(target, dtype=torch.long)
