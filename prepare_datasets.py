import chess_seq.data.preprocessing as preprocessing
from chess_seq.encoder import MoveEncoder

import argparse

encoder = MoveEncoder()
encoder.load("data/move_encoder.pkl")

parser = argparse.ArgumentParser()
parser.add_argument("--pgn_in", type=str, default="")
parser.add_argument("--npz_out", type=str, default="")
args = parser.parse_args()

input_folder = args.pgn_in
output_folder = args.npz_out

preprocessing.run_through_folder(input_folder, output_folder, encoder)
