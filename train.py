import torch

from configs import TrainingSession, ModelConfig, TrainingConfig
from chess_seq import ChessTrainerRunner


session_config = TrainingSession()

runner = ChessTrainerRunner(
    session_config=session_config,
    model_config=ModelConfig(),
    training_config=TrainingConfig(),
)

if session_config.compile:
    runner.model = torch.compile(runner.model)

runner.train()
