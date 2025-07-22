from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "robo_chuk"
    vocab_size: int = 71
    block_size: int = 2048
    n_head: int = 4
    n_layers: int = 3
    dropout: int = 0.1
    k: int = 128  # k needs to be divisible by n_head


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    lr_min: float = 1e-6
    batch_size: int = 16
    warmup: float = 1000
    final_lr_time: float = 100000

    optimizer: str = "adam"
    scheduler: str = "warmup_cosine"


@dataclass
class TrainingSession:
    new_model: bool = False
    model_name: str = "robo_chuk"
    data_folder: str = "data/gms_csvs/"
    encoder_path: str = "data/move_encoder.pkl"

    num_epochs: int = 10
    test_interval: int = 100
    checkpoint_interval: int = 200
    test_games_lengths: List[int] = field(default_factory=lambda: [30])

    change_config: bool = False
    batch_size: int = 16
    lr: int = 1e-4 * batch_size / 16
    lr_min = 1e-6
    warmup = 1000
    final_lr_time = 100000
