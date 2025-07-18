import os
import pickle

import chess_seq.chess_utils as chess_utils

import argparse

with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--pgn_in", type=str, default="")
parser.add_argument("--csv_out", type=str, default="")
args = parser.parse_args()

input_folder = args[0]
output_folder = args[1]

# for input_file in os.listdir(input_folder):
#     print(f"Processing {input_file}...")
#     if input_file.endswith(".pgn"):
#         input_path = os.path.join(input_folder, input_file)
#         output_path = os.path.join(output_folder, input_file.replace(".pgn", ".csv"))
#         chess_utils.process_pgn_file(
#             input_path, output_path, end_id=None, encoder=encoder
#         )
