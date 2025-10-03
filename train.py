from configs import TrainingSession
from chess_seq.training.trainer_runner import ChessTrainerRunner
from chess_seq.tictactoe.trainer import TTTTrainerRunner


# runner = ChessTrainerRunner(session_config=TrainingSession())
runner = TTTTrainerRunner(session_config=TrainingSession())

runner.train()
