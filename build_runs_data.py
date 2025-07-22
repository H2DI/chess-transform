import chess_seq.utils as utils
from chess_seq.training.generate_runs import generate_runs


OUTPUT_DIR = "synthetic_games"
NEW_GAMES = 4096
N_PLIES = 100

MODEL_NAME = "vasyl_k128_n4_h4"

model, encoder, checkpoint = utils.load_model(MODEL_NAME)
n_games = checkpoint["n_games"]

generate_runs(NEW_GAMES, model, MODEL_NAME, n_games, OUTPUT_DIR, encoder, N_PLIES)
