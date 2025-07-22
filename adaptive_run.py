from configs import TrainingSession
from chess_seq.training.trainer_runner import ChessTrainerRunner

from chess_seq.training.generate_runs import generate_runs
from publish_game import publish_game


N_ITERATIONS = 60  # Number of training iterations

OUTPUT_DIR = "synthetic_games/"
NEW_GAMES = 128
N_PLIES = 100
STUDY_ID = "LdUHTfjo"


session_config = TrainingSession()
runner = ChessTrainerRunner(session_config=session_config)
model_name = session_config.model_name

for _ in range(N_ITERATIONS):
    runner.train(skip_seen_files=False)

    model = runner.model
    n_games = runner.n_games
    encoder = runner.encoder

    publish_game(model_name, STUDY_ID)
    generate_runs(NEW_GAMES, model, model_name, n_games, OUTPUT_DIR, encoder, N_PLIES)
