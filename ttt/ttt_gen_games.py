import sys
import os
import pickle
import chess_seq.tictactoe.mechanics as mechanics
import chess_seq.tictactoe.players as players


encoder_path = "data/ttt_encoder.pkl"


if __name__ == "__main__":
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
        game = mechanics.TTTBoard()
        random_player = players.RandomPlayer(game)
        recorder = players.GamesRecorder(encoder, random_player)

        if len(sys.argv) > 1:
            N = int(sys.argv[1])
            for _ in range(N):
                recorder.play_game()

            save_dir = "synthetic_games"
            base_filename = "ttt_random_games"
            ext = ".npz"
            filepath = os.path.join(save_dir, base_filename + ext)
            counter = 1

            # Find a new filename if file already exists
            while os.path.exists(filepath):
                filepath = os.path.join(save_dir, f"{base_filename}_{counter}{ext}")
                counter += 1

            recorder.save_games(filepath)

            print(f"Saved {N} games to {filepath}")
            for _ in range(5):
                recorder.play_game(print_result=True)

        else:
            print("No argument provided.")
