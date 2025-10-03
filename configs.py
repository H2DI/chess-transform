from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "ttt_deep_large"
    vocab_size: int = 14
    block_size: int = 12
    n_head: int = 4
    n_layers: int = 6
    dropout: int = 0.1
    k: int = 128  # k needs to be divisible by n_head

    special_freqs: List[float] = field(default_factory=lambda: [2 * 3.14159 / 2])
    encoder_path: str = "data/ttt_encoder.pkl"


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    lr_min: float = 1e-5
    batch_size: int = 64
    warmup: float = 1000
    final_lr_time: float = 10_000

    optimizer: str = "adam"
    scheduler: str = "warmup_cosine"


@dataclass
class TrainingSession:
    new_model: bool = False
    model_name: str = "ttt_deep_large"
    data_folder: str = "synthetic_games/"

    data_format: str = "npz"  # or "csv"
    device_str: str = "cpu"

    num_epochs: int = 10
    test_interval: int = 512
    checkpoint_interval: int = 4096
    test_games_lengths: List[int] = field(default_factory=lambda: [13])

    restart: bool = True
    change_config: bool = False
    # batch_size: int = 16
    # lr: int = 1e-3 * batch_size / 16
    # lr_min = 1e-5
    # warmup = 1000
    # final_lr_time = 100000


@dataclass
class RLTraining:
    model_name: str = "ttt_large"
    log_dir: str = "logs/rl_training"
