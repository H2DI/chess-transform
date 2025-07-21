from chess_seq.evaluation.testing_model import play_valid_game
import utils
import chess

import json

import requests

with open("private_token.json") as f:
    token = json.load(f)["lichessApiToken"]

MODEL_NAME = "vasyl_k64_n4_h4"
study_id = "ZB0upGx"
study_id = "jGATtknM"

model, encoder, checkpoint = utils.load_model(MODEL_NAME)
n_games = checkpoint["n_games"]

game = chess.Board()
game, pgn, bad_plies = play_valid_game(model, encoder, game=game, n_plies=200)
print(
    f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
    if bad_plies
    else "0 bad moves"
)
# print(pgn.mainline_moves())

# Using requests
headers = {"Authorization": f"Bearer {token}"}
res = requests.get(f"https://lichess.org/api/study/{study_id}.pgn", headers=headers)
pgn_text = res.text

# Count chapters by number of "[Event " tags
num_chapters = pgn_text.count("[Event ")

print(f"https://lichess.org/study/{study_id}")
if num_chapters == 64:
    Exception("Study is full with 64 chapters.")
else:
    print(f"Study can have {64 - num_chapters} more chapters, adding a new one.")

res = requests.post(
    f"https://lichess.org/api/study/{study_id}/import-pgn",
    headers=headers,
    data={
        "name": f"{MODEL_NAME}_{str(n_games)}",
        "pgn": pgn,
        "orientation": "white",
    },
)

if res.status_code == 200:
    print("Game successfully published to the study.")
