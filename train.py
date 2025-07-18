from tqdm import tqdm
import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


import chess_seq.models as models
import chess_seq.datasets as datasets
import chess_seq.testing_model as testing_model
import utils

model_name = "bob"

N_VOCAB = 71
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 1e-4

######################################## DATA ###############################
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
device = torch.device("cpu")


def collate_fn(batch_list):
    inputs_padded = pad_sequence(batch_list, batch_first=True, padding_value=N_VOCAB)
    return inputs_padded


csv_train = (
    "/Users/hadiji/Documents/GitHub/chess-transform/"
    + "data/train_csvs/lichess_elite_"
    + "2019-02.csv"
)

dataset = datasets.ChessDataset(csv_train, device=device)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, pin_memory=False
)
with open(csv_train, "r") as f:
    num_lines = sum(1 for _ in f)
print(f"Number of games in training CSV: {num_lines}")

######################### MODEL ##############################

checkpoint_path = utils.get_latest_checkpoint(model_name)

state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
model_config = state_dict["model_config"]

model = models.ChessNet(config=model_config).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# warmup then cosing annealing
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


n_steps = state_dict["n_steps"]
n_games = state_dict["n_games"]
encoder = state_dict["encoder"]

model.load_state_dict(state_dict["model_state_dict"])
optimizer.load_state_dict(state_dict["optimizer_state_dict"])
scheduler.load_state_dict(state_dict["scheduler_state_dict"])


criterion = nn.CrossEntropyLoss(ignore_index=N_VOCAB)
writer = SummaryWriter(log_dir=f"runs/chess_transformer_experiment/{model_config.name}")


for epoch in range(NUM_EPOCHS):
    model.train()
    print("Start training")
    for i, seq in tqdm(enumerate(dataloader)):
        n_steps += 1
        n_games += BATCH_SIZE
        seq = seq.to(device)

        input_seq = seq[:, :-1]
        target = seq[:, 1:]

        b, T = seq.shape

        tgt_mask = torch.tril(torch.ones(T - 1, T - 1)).to(device).bool()
        logits = model(input_seq, mask=tgt_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))

        with torch.no_grad():
            writer.add_scalar("Loss/train", loss.item(), n_steps)
            writer.add_scalar(
                "Loss/train-log", torch.log(loss).item(), math.log(n_steps)
            )
            writer.add_scalar("LR", scheduler.get_last_lr()[0], n_steps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm**0.5
            writer.add_scalar("GradNorm/train", grad_norm, n_steps)
            with torch.no_grad():
                weight_norm = 0.0
                for param in model.parameters():
                    weight_norm += param.data.norm(2).item() ** 2
                weight_norm = weight_norm**0.5
                writer.add_scalar("WeightNorm/train", weight_norm, n_steps)

        if i % 300 == 0:
            model.eval()
            testing_model.check_games(model, encoder)
            model.train()
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

print("Training complete")
model.eval()
testing_model.check_games(model, encoder)
model.train()
