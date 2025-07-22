from configs import TrainingSession
from chess_seq.training.trainer_runner import ChessTrainerRunner


runner = ChessTrainerRunner(session_config=TrainingSession())

runner.train()
