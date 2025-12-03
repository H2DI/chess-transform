from configs import TrainingSession, ModelConfig, TrainingConfig
from chess_seq import ChessTrainerRunner
import torch

torch.set_float32_matmul_precision("high")  # or "medium"

session_config = TrainingSession()

runner = ChessTrainerRunner(
    session_config=session_config,
    model_config=ModelConfig(),
    training_config=TrainingConfig(),
)


runner.train()
