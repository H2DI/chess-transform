from chess_seq.testing_model import play_valid_game
import utils
import chess

import json

import requests

with open("private_token.json") as f:
    token = json.load(f)["lichessApiToken"]

headers = {"Authorization": f"Bearer {token}"}

model_name = "bob"

model, encoder, checkpoint = utils.load_model(model_name)

n_games = checkpoint["n_games"]

game = chess.Board()

game, pgn, bad_plies = play_valid_game(model, encoder, game=game, n_plies=200)
print(
    f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
    if bad_plies
    else "0 bad moves"
)
print(pgn.mainline_moves())

study_id = "ZB0upGxH"

res = requests.post(
    f"https://lichess.org/api/study/{study_id}/import-pgn",
    headers=headers,
    data={
        "name": model_name + str(n_games),
        "pgn": pgn,
        "orientation": "white",
    },
)
