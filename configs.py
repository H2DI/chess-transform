from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str = "ttt_tiny"
    vocab_size: int = 14
    block_size: int = 12
    n_head: int = 4
    n_layers: int = 3
    dropout: int = 0.0
    k: int = 8  # k needs to be divisible by n_head

    special_freqs: List[float] = field(default_factory=lambda: [2 * 3.14159 / 2])
    encoder_path: str = "data/ttt_encoder.pkl"


@dataclass
class GRPOConfig:
    new_model: bool = False

    model_name: str = "ttt_small_zero"
    log_dir: str = "logs/rl_training"
    device_str: str = "cpu"
    eval_frequency: int = 1024
    max_episodes = 500000

    agent_start = None  # None means random start
    p_start = 0.5  # probability of agent starting first in evaluation games

    beta = 0.02
    epsilon_low = 0.1
    epsilon_high = 0.3

    group_size = 2
    groups_between_prompts = 2
    prompts_between_models = 2

    rollout_temperature = 1

    learning_rate = 1e-3
    min_lr = 1e-4
    end_lr_steps = 50000

    debug_prints = True


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
    new_model: bool = True
    model_name: str = "ttt_small_zero"
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
