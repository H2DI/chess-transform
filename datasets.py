import torch
from torch.utils.data import Dataset
import pickle

import chess
import chess.pgn

# import time


def parse_pgn(game, encoder):
    board = game.board()
    moves = []
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    return encoder.transform(moves)


def generate_games_list():
    # a = time.time()
    with open("data/move_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    # print(time.time() - a)
    parsed_games = []
    with open("data/lichess_elite_2024-06.pgn") as pgn:
        for _ in range(100):
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
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )
