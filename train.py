from config import TrainingSession, ModelConfig, TrainingConfig
from chess_seq.training.trainer_runner import ChessTrainerRunner
from chess_seq.tictactoe.trainer import TTTTrainerRunner


# runner = ChessTrainerRunner(session_config=TrainingSession())
runner = TTTTrainerRunner(
    session_config=TrainingSession(),
    model_config=ModelConfig(),
    training_config=TrainingConfig(),
)
# runner.train()
