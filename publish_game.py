from chess_seq.testing_model import play_valid_game
import utils
import chess
import requests
import json

with open("private_token.json") as f:
    token = json.load(f)["lichessApiToken"]

headers = {"Authorization": f"Bearer {token}"}

model_name = "bob"

model, encoder, _ = utils.load_model(model_name)

game = chess.Board()

game, pgn, bad_plies = play_valid_game(model, encoder, game=game, n_plies=150)
print(
    f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
    if bad_plies
    else "0 bad moves"
)
print(pgn.mainline_moves())

res = requests.post(
    "https://lichess.org/api/import", data={"pgn": pgn}, headers=headers
)

game_id = res.json()["id"]
print("Uploaded game:", game_id)
