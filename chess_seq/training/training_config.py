from dataclasses import dataclass


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    lr_min: float = 1e-6
    batch_size: int = 16
    warmup: float = 1000
    final_lr_time: float = 100000

    optimizer: str = "adam"
    scheduler: str = "warmup_cosine"
