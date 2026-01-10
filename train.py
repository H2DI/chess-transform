import torch

from configs import TrainingSession, ModelConfig, TrainingConfig
from chess_seq import ChessTrainerRunner

torch.set_float32_matmul_precision("high")  # or "medium"

runner = ChessTrainerRunner(
    session_config=TrainingSession(),
    model_config=ModelConfig(),
    training_config=TrainingConfig(),
)


runner.train()
