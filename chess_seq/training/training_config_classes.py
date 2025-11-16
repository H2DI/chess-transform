from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    lr_min: float = 1e-6
    batch_size: int = 16
    warmup: float = 1000
    final_lr_time: float = 100000

    optimizer: str = "adam"
    scheduler: str = "warmup_cosine"


@dataclass
class TrainingSession:
    model_name: str
    data_folder: str
    encoder_path: str

    new_model: bool
    num_epochs: int
    test_interval: int
    checkpoint_interval: int
    test_games_lengths: List[int]
