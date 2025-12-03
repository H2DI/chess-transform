from configs import TrainingSession, ModelConfig, TrainingConfig
from chess_seq import ChessTrainerRunner


runner = ChessTrainerRunner(
    session_config=TrainingSession(),
    model_config=ModelConfig(),
    training_config=TrainingConfig(),
)
runner.train()
