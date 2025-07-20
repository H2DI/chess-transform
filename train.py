from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import chess_seq.models as models
import chess_seq.training as training
import chess_seq.datasets as datasets
import chess_seq.testing_model as testing_model
import utils
import os

model_name = "sarah"

CHANGE_CONFIG = False

# Default behavior is not use the following parameters, unless CHANGE_CONFIG is True
BATCH_SIZE = 16
LR = 1e-4 * BATCH_SIZE / 16
LR_MIN = 1e-6
NUM_EPOCHS = 3
WARMUP = 1000
FINAL_LR_TIME = 100000

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
device = torch.device("cpu")


#### Data

csv_folder = "/Users/hadiji/Documents/GitHub/chess-transform/data/train_csvs/"
csv_files = [csv_folder + f for f in os.listdir(csv_folder) if f.endswith(".csv")]


#### Load model and training configs

checkpoint_path = utils.get_latest_checkpoint(model_name)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model_config = checkpoint["model_config"]
training_config = checkpoint["training_config"]

if CHANGE_CONFIG:
    training_config.lr = LR
    training_config.lr_min = LR_MIN
    training_config.batch_size = BATCH_SIZE
    training_config.warmup = WARMUP
    training_config.final_lr_time = FINAL_LR_TIME


model = models.ChessNet(config=model_config).to(device)

optimizer, scheduler = training.initialize_optimizer(training_config, model)

n_steps = checkpoint["n_steps"]
n_games = checkpoint["n_games"]
encoder = checkpoint["encoder"]

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


n_vocab = model_config.vocab_size
criterion = nn.CrossEntropyLoss(ignore_index=n_vocab)
writer = SummaryWriter(log_dir=f"runs/chess_transformer_experiment/{model_config.name}")


for file_number, csv_train in enumerate(csv_files):
    if file_number <= checkpoint["file_number"]:
        print(f"Skipping {csv_train} as it is already processed.")
        continue

    print(f"File : {csv_train}")
    # Load data

    dataloader = datasets.build_dataloader(
        csv_train,
        batch_size=training_config.batch_size,
        device=device,
        padding_value=n_vocab,
    )

    with open(csv_train, "r") as f:
        num_lines = sum(1 for _ in f)
    print(f"Number of games in training CSV: {num_lines}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        print("Start training")
        for i, seq in tqdm(enumerate(dataloader)):
            n_steps += 1
            n_games += training_config.batch_size

            loss, logits = training.train_step(seq, model, criterion, device)

            with torch.no_grad():
                writer.add_scalar("Loss/train", loss.item(), n_steps)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], n_steps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                training.log_grads(writer, model, n_steps)
                training.log_weight_norms(writer, model, n_steps)

            if i % 250 == 0:
                model.eval()
                testing_model.check_games(model, encoder)

                n_bad, t_first_bad = testing_model.test_first_moves(model, encoder)
                training.log_stat_group(writer, "Play/NumberOfBadMoves", n_bad, n_games)
                training.log_stat_group(
                    writer, "Play/FirstBadMoves", t_first_bad, n_games
                )

                model.train()

            if i % 1000 == 0:
                checkpoint = {
                    "model_config": model_config,
                    "training_config": training_config,
                    "n_steps": n_steps,
                    "n_games": n_games,
                    "file_number": file_number,
                    "encoder": encoder,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                training.save_checkpoint(checkpoint)

model.eval()
testing_model.check_games(model, encoder)
model.train()
