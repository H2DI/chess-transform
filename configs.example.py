"""
Example configuration file for training.

Copy this file to `configs.py` and modify as needed.
The actual configs.py is gitignored to allow local customization.
"""

from chess_seq.configs import ModelConfig, TrainingConfig, TrainingSession, GRPOConfig

# Model configuration - default is the 450M parameter model
model_config = ModelConfig(
    name="my_chess_model",
    vocab_size=4611,
    block_size=256,
    k=1024,
    head_dim=128,
    n_head=16,
    n_layers=28,
    dropout=0.0,
    kv_groups=2,
)

# Training hyperparameters
training_config = TrainingConfig(
    lr=1e-4,
    lr_min=1e-6,
    wd=0.01,
    batch_size=64,
    num_epochs=2,
)

# Training session settings
session_config = TrainingSession(
    resume=False,
    model_name="my_chess_model",
    data_folder="data/train_npz/",
    device_str="cuda",  # Change to "cpu" if no GPU
    test_interval=512,
)

# GRPO reinforcement learning config (experimental)
grpo_config = GRPOConfig(
    model_name="my_chess_model",
    device_str="cuda",
    learning_rate=1e-4,
)
