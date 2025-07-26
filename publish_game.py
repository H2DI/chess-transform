import json
import requests

from chess_seq.evaluation.game_engine import ChessGameEngine
import chess_seq.utils as utils
import chess

import configs


with open("private_token.json") as f:
    token = json.load(f)["lichessApiToken"]

config = configs.TrainingSession()
model_name = config.model_name

study_id = "ZB0upGx"
study_id = "jGATtknM"
study_id = "LdUHTfjo"  # ada_chuk


def publish_game(model_name, study_id):
    model, encoder, checkpoint = utils.load_model(model_name)
    n_games = checkpoint["n_games"]

    game = chess.Board()
    engine = ChessGameEngine(model, encoder)

    game, pgn, bad_plies = engine.play_game(game=game, n_plies=200, greedy=False)
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
            "name": f"{model_name}_{str(n_games)}",
            "pgn": pgn,
            "orientation": "white",
        },
    )

    if res.status_code == 200:
        print("Game successfully published to the study.")


if __name__ == "__main__":
    model, encoder, checkpoint = utils.load_model(model_name)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")

    publish_game(model_name, study_id)
