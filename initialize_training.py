import pickle

from torch.utils.tensorboard import SummaryWriter

import chess_seq.models as models
import chess_seq.testing_model as testing_model
import chess_seq.training as training
import utils

with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

model_name = "sarah"
model_config = models.ModelConfig(name=model_name, n_layers=4, n_head=4, k=128)
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

training_config = training.TrainingConfig(
    lr=lr,
    lr_min=lr_min,
    warmup=warmup,
    final_lr_time=final_lr_time,
    batch_size=batch_size,
)


model = models.ChessNet(config=model_config)

optimizer, scheduler = training.initialize_optimizer(training_config, model)

writer = SummaryWriter(log_dir=f"runs/{model_config.name}")

checkpoint = {
    "model_config": model_config,
    "training_config": training_config,
    "file_number": 0,
    "n_steps": 0,
    "n_games": 0,
    "encoder": encoder,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}


training.save_checkpoint(checkpoint)

model.eval()
testing_model.check_games(model, encoder)
model.train()
