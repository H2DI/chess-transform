import pickle
import chess_seq.data.preprocessing as preprocessing

import argparse

with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--pgn_in", type=str, default="")
parser.add_argument("--csv_out", type=str, default="")
args = parser.parse_args()

input_folder = args[0]
output_folder = args[1]

preprocessing.run_through_folder(input_folder, output_folder, encoder)
