from evaluation.testing_model import play_valid_game
import csv
import os


def generate_runs(N, model, model_name, n_games, output_dir, encoder):
    csv_path = os.path.join(output_dir, f"{model_name}_games.csv")
    fieldnames = ["game_index", "pgn", "bad_plies"]

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    fieldnames = ["game_index", "pgn", "bad_plies"]

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(N):
            game, pgn, bad_plies = play_valid_game(model, encoder, n_plies=30)
            writer.writerow({"game_index": i, "pgn": pgn, "bad_plies": bad_plies})
    for _ in range(N):
        game, pgn, bad_plies = play_valid_game(model, encoder, n_plies=30)
