import torch
import torch.nn as nn
import torch.optim as optim

import chess_transformer
import datasets

import testing_model

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

import pickle

with open("data/move_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
N_VOCAB = len(encoder.classes_)


N_VOCAB = 36727
BATCH_SIZE = 64
NUM_EPOCHS = 2
LOAD = False
LR = 1e-3


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=N_VOCAB)
    targets = torch.stack(targets)
    lengths = torch.tensor([len(input_) for input_ in inputs])
    return inputs_padded, targets, lengths


games_list = datasets.generate_games_list(first_game=0, n_games=32)
dataset = datasets.NextMoveDatasetPGN(games_list)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)


games_list = datasets.generate_games_list(first_game=100, n_games=10)
validation_set = datasets.NextMoveDatasetPGN(games_list)
validation_dataloader = DataLoader(
    validation_set, batch_size=32, shuffle=False, collate_fn=collate_fn
)

model = chess_transformer.ChessNet2(n_vocab=N_VOCAB)
if LOAD:
    state_dict = torch.load("models/chess-transform2.pth")
    model.load_state_dict(state_dict)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)


def create_padding_mask(sequence_lengths):
    batch_size = sequence_lengths.size(0)
    max_length = sequence_lengths.max()
    indices = torch.arange(max_length).expand(batch_size, max_length)
    mask = indices < sequence_lengths.unsqueeze(1)
    return mask


print(len(dataset))

for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    print("Start training")
    for i, (input_seq, target_move, lengths) in tqdm(enumerate(dataloader)):
        mask = create_padding_mask(lengths)
        logits = model(input_seq, mask=mask)
        loss = criterion(logits, target_move)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 20 == 0:
            print(testing_model.finish_game(model, encoder))

    for input_seq, target_move, lengths in validation_dataloader:
        with torch.no_grad():
            mask = create_padding_mask(lengths)
            logits = model(input_seq, mask=mask)
            val_loss = criterion(logits, target_move)

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {val_loss:.4f}")
    torch.save(model.state_dict(), "models/chess-transform2.pth")
