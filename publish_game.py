import json
import requests
import torch

from chess_seq import ChessGameEngine, MoveEncoder
import chess_seq.utils as utils
import chess

import configs


with open("private_token.json") as f:
    token = json.load(f)["lichessApiToken"]

config = configs.ModelConfig()
model_name = config.name

study_id = "ZB0upGx"
study_id = "jGATtknM"
study_id = "LdUHTfjo"  # ada_chuk

study_id = "ZbXAbPvL"


def publish_game(model_name, study_id):
    model, _, checkpoint = utils.load_model(model_name)
    model.to(torch.device("cpu"))
    n_games = checkpoint["n_games"]

    encoder = MoveEncoder()
    encoder.load(config.encoder_path)

    engine = ChessGameEngine(model, encoder)
    game = chess.Board()
    game, pgn, bad_plies = engine.play_game(game=game, n_plies=200, greedy=False)
    num_plies = len(game.move_stack)

    comment = (
        f"{len(bad_plies)} bad moves out of {num_plies}. First bad ply: {bad_plies[0]}"
        if bad_plies
        else "0 bad moves"
    )
    pgn.comment = comment
    print(comment)
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
    publish_game(model_name, study_id)
