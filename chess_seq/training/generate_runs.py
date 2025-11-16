from chess_seq.game_engine import ChessGameEngine

import numpy as np
import os

"""
TODO: 
manage the game seeds
"""


def open_npz_dir(output_dir, model_name, n_games):
    # final directory: data/<output_dir>/<model>_<n_games>_games/
    base_dir = os.path.join("data", output_dir, f"{model_name}_{n_games}_games")

    os.makedirs(base_dir, exist_ok=True)

    # find last game_id
    existing = [f for f in os.listdir(base_dir) if f.endswith(".npz")]
    if not existing:
        last_game_id = 0
    else:
        ids = [int(f.replace(".npz", "")) for f in existing]
        last_game_id = max(ids)

    return last_game_id, base_dir


def generate_runs_npz(N, model, model_name, n_games, output_dir, encoder, n_plies):
    engine = ChessGameEngine(model, encoder)
    game_id, base_dir = open_npz_dir(output_dir, model_name, n_games)

    for _ in range(N):
        game, _, _ = engine.play_game(n_plies=n_plies, record_pgn=False)
        tokens = np.array(encoder.board_to_sequence(game), dtype=np.int32)

        game_id += 1
        out_path = os.path.join(base_dir, f"{game_id}.npz")

        np.savez_compressed(out_path, game_id=game_id, tokens=tokens)

        print(f"Saved game {game_id} to {out_path}")
