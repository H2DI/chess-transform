import pickle

from torch.utils.tensorboard import SummaryWriter

import chess_seq.models as models
import chess_seq.evaluation.testing_model as testing_model
import chess_seq.training.trainer as trainer

from chess_seq.training.training_config import TrainingConfig

with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

n_layers = 2
n_head = 4
k = 128
model_name = f"vasyl_k{k}_n{n_head}_h{n_head}"

assert k % n_head == 0, "k must be divisible by n_head"

model_config = models.ModelConfig(
    name=model_name, n_layers=n_layers, n_head=n_head, k=k
)
# Default values
# name = "Default"
# vocab_size = 71
# block_size = 2048
# n_head = 4
# n_layers = 2
# dropout = 0.1
# k = 64  # k needs to be divisible by n_head


# Training parameters
lr = 1e-4
lr_min = 1e-6
batch_size = 16

warmup = 1000
final_lr_time = 100000

training_config = TrainingConfig(
    lr=lr,
    lr_min=lr_min,
    warmup=warmup,
    final_lr_time=final_lr_time,
    batch_size=batch_size,
)


model = models.ChessNet(config=model_config)

optimizer, scheduler = trainer.initialize_optimizer(training_config, model)

writer = SummaryWriter(log_dir=f"runs/chess_transformer_experiment/{model_config.name}")


trainer.save_checkpoint(
    model_config,
    training_config,
    0,
    0,
    0,
    encoder,
    model,
    optimizer,
    scheduler,
)

model.eval()
testing_model.print_basic_games(model, encoder)
model.train()

print(f"{model_name=}")
