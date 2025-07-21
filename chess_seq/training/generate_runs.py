from chess_seq.chess.chess_utils import board_to_sequence
from chess_seq.evaluation.game_engine import ChessGameEngine

import csv
import os

"""
TODO: 
manage the game seeds
"""


def generate_runs(N, model, model_name, n_games, output_dir, encoder):
    output_file = None
    engine = ChessGameEngine(model, encoder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{model_name}_{n_games}_games.csv")
    if os.path.exists(output_file):
        with open(output_file, "r", newline="") as infile:
            reader = csv.reader(infile)
            next(reader, None)  # skip header
            rows = list(reader)
            if rows:
                game_id = int(rows[-1][0])
            else:
                game_id = 0
    else:
        with open(output_file, "w", newline="") as out:
            csv_writer = csv.writer(out)
            csv_writer.writerow(["game_id", "tokens"])
            game_id = 0

    with open(output_file, "a", newline="") as out:
        csv_writer = csv.writer(out)
        for _ in range(N):
            game, _, _ = engine.play_game(n_plies=30, record_pgn=False)
            encoded_moves = board_to_sequence(game, encoder)
            game_id += 1
            csv_writer.writerow(
                [game_id, " ".join([str(move) for move in encoded_moves])]
            )
