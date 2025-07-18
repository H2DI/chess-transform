import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


import chess_seq.models as models
import chess_seq.testing_model as testing_model
import utils


with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


model_config = models.ModelConfig(name="mike", n_layers=6, n_head=4)
# name = "Default"
# vocab_size = 71
# block_size = 2048
# n_head = 4
# n_layers = 2
# dropout = 0.1
# k = 64  # k needs to be divisible by n_head

model = models.ChessNet(config=model_config)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=1000
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=99_000, eta_min=1e-6
        ),
    ],
    milestones=[1000],
)

n_steps = 0
n_games = 0

writer = SummaryWriter(log_dir=f"runs/chess_transformer_experiment/{model_config.name}")

checkpoint = {
    "model_config": model_config,
    "n_steps": n_steps,
    "n_games": n_games,
    "encoder": encoder,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
}


utils.save_checkpoint(checkpoint)

model.eval()
testing_model.check_games(model, encoder)
model.train()
