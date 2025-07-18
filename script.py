from chess_seq.testing_model import play_valid_game, test_first_moves
import chess_seq.models as models
import utils
import torch
import chess

# import os
# import re


model_name = "john"

checkpoint_path = utils.get_latest_checkpoint(model_name)
checkpoint = torch.load(
    checkpoint_path, map_location=torch.device("cpu"), weights_only=False
)

model_config = checkpoint["model_config"]
model = models.ChessNet(model_config)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

encoder = checkpoint["encoder"]

game = chess.Board()
game.push(chess.Move.from_uci("g1f3"))

# e4
# game.push(chess.Move.from_uci("e2e4"))

# d4
# game.push(chess.Move.from_uci("d2d4"))

# c4
# game.push(chess.Move.from_uci("c2c4"))

# a4
# game.push(chess.Move.from_uci("a2a4"))

# f3
# game.push(chess.Move.from_uci("f2f3"))

# Nh3
# game.push(chess.Move.from_uci("g1h3"))

# French
# game.push(chess.Move.from_uci("e2e4"))
# game.push(chess.Move.from_uci("e7e6"))

# Sicilian
# game.push(chess.Move.from_uci("e2e4"))
# game.push(chess.Move.from_uci("c7c5"))

# e4 e5
# game.push(chess.Move.from_uci("e2e4"))
# game.push(chess.Move.from_uci("e7e5"))

# d4 d5
# game.push(chess.Move.from_uci("d2d4"))
# game.push(chess.Move.from_uci("d7d5"))


game, pgn, bad_plies = play_valid_game(model, encoder, game=game, n_plies=150)
print(
    f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
    if bad_plies
    else "0 bad moves"
)
print(pgn.mainline_moves())

n_bad, t_first_bad = test_first_moves(model, encoder, prints=True)
