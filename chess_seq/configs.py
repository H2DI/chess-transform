from dataclasses import dataclass, field, asdict
from typing import List

import json


@dataclass
class Config:
    def to_dict(self):
        return asdict(self)

    def save(self) -> str:
        with open(f"configs/{self.name}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ModelConfig(Config):
    name: str = "default"
    vocab_size: int = 4611
    block_size: int = 256
    k: int = 1024
    head_dim: int = 128
    n_head: int = 16
    n_layers: int = 28
    dropout: float = 0.0
    kv_groups: int = 2
    ff_expansion: int = 3

    pad_index: int = 4610
    special_freqs: List[float] = field(default_factory=lambda: [2])
    encoder_path: str = "data/id_to_token.json"


@dataclass
class TrainingConfig(Config):
    lr: float = 1e-4
    lr_min: float = 1e-6
    wd: float = 0.01

    batch_size: int = 64
    total_games: int = 12_421_396

    warmup: float = 0.01
    num_epochs: int = 2


@dataclass
class TrainingSession(Config):
    resume: bool = False
    jit: bool = False

    model_name: str = "gamba_rossa"
    data_folder: str = "data/train_npz/"

    device_str: str = "cpu"

    test_interval: int = 512
    checkpoint_interval: int = 13_000_000
    test_games_lengths: List[int] = field(default_factory=lambda: [13])


@dataclass
class GRPOConfig(Config):
    new_model: bool = False

    model_name: str = "ttt_large"
    log_dir: str = "logs/rl_training/"
    device_str: str = "cpu"
    eval_frequency: int = 1024
    max_episodes = 500000

    agent_start = None  # None means random start
    p_start = 0.5  # probability of agent starting first in evaluation games

    beta = 0.02
    epsilon_low = 0.1
    epsilon_high = 0.3

    group_size = 128
    groups_between_prompts = 2
    prompts_between_models = 2

    rollout_temperature = 1

    learning_rate = 1e-4
    min_lr = 1e-5
    end_lr_steps = 50000

    debug_prints = False
