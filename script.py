from chess_seq.testing_model import play_valid_game, test_first_moves
import utils
import chess

# import os
# import re


model_name = "bob"
number = "3561008"
model, encoder, checkpoint = utils.load_model(model_name, number=number)


game = chess.Board()
# game.push(chess.Move.from_uci("g1f3"))

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
# print(pgn)
print(pgn.mainline_moves())

n_bad, t_first_bad = test_first_moves(model, encoder, prints=True)
