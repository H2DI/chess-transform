import chess_seq.utils as utils
from chess_seq.training.generate_runs import generate_runs


output_dir = "synthetic_games"

model_name = "mike"
model, encoder, checkpoint = utils.load_model(model_name)
n_games = checkpoint["n_games"]


generate_runs(3000, model, model_name, n_games, output_dir, encoder)
