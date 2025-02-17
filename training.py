import torch
import torch.nn as nn
import torch.optim as optim

import chess_transformer
import datasets

import testing_model

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


import pickle

with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


N_VOCAB = 66

BATCH_SIZE = 64
NUM_EPOCHS = 4
LOAD = True
LR = 1e-3


def collate_fn(batch_list):
    inputs_padded = pad_sequence(batch_list, batch_first=True, padding_value=N_VOCAB)
    return inputs_padded


csv_train = "data/lichess_elite_2024-06_train.csv"
dataset = datasets.ChessDataset(csv_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


csv_val = "data/lichess_elite_2024-06_val.csv"
val_set = datasets.ChessDataset(csv_val)
validation_dataloader = DataLoader(val_set, batch_size=32, collate_fn=collate_fn)

model = chess_transformer.ChessNet(config=chess_transformer.ModelConfig)
if LOAD:
    state_dict = torch.load("models/chess-transform2.pth")
    model.load_state_dict(state_dict)


criterion = nn.CrossEntropyLoss(ignore_index=N_VOCAB)
optimizer = optim.SGD(model.parameters(), lr=LR)


for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    print("Start training")
    for i, seq in enumerate(dataloader):
        input_seq = seq[:, :-1]
        target = seq[:, 1:]
        logits = model(input_seq)
        loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 100 == 0:
            game = testing_model.finish_game(model)
            print(encoder.inverse_transform(game))

    avg_loss = total_loss / i
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    # print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), "models/chess-transform2.pth")
