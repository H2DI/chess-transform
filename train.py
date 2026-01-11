import torch

from chess_seq import ChessTrainerRunner
from chess_seq.configs import TrainingSession, ModelConfig, TrainingConfig


MODEL_NAME = "gamba_gialla"

training_sess = TrainingSession(
    resume=True,
    jit=False,
    model_name=MODEL_NAME,
    data_folder="data/train_npz/",
    device_str="cpu",
    test_interval=512,
    checkpoint_interval=13_000_000,
    test_games_lengths=[13],
)


training_cfg = TrainingConfig(
    lr=1e-4,
    lr_min=1e-6,
    wd=0.01,
    batch_size=64,
    total_games=12_421_396,
    warmup=0.01,
    num_epochs=2,
)

model_cfg = ModelConfig.from_json_file("configs/{MODEL_NAME}.json")

torch.set_float32_matmul_precision("high")

runner = ChessTrainerRunner(
    session_config=training_sess,
    model_config=model_cfg,
    training_config=training_cfg,
)


runner.train()
