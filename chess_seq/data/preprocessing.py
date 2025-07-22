import csv
import chess.pgn
import os

import chess_seq.chess_utils.chess_utils as chess_utils


def process_pgn_file(input_file, output_file, encoder, end_id=None, start_id=0):
    with open(input_file, "r") as pgn, open(output_file, "w", newline="") as out:
        csv_writer = csv.writer(out)
        csv_writer.writerow(["game_id", "tokens"])

        game_id = start_id
        while True:
            if end_id is not None and game_id >= end_id:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            encoded_moves = chess_utils.pgn_to_sequence(game, encoder)
            csv_writer.writerow(
                [game_id, " ".join([str(move) for move in encoded_moves])]
            )
            game_id += 1


def run_through_folder(input_folder, output_folder, encoder):
    for input_file in os.listdir(input_folder):
        print(f"Processing {input_file}...")
        if input_file.endswith(".pgn"):
            input_path = os.path.join(input_folder, input_file)
            output_path = os.path.join(
                output_folder, input_file.replace(".pgn", ".csv")
            )
            process_pgn_file(input_path, output_path, end_id=None, encoder=encoder)
