import torch

import chess_transformer
import pickle

import datasets
import matplotlib.pyplot as plt
import numpy as np

N_VOCAB = 66

# model = chess_transformer.ChessNet2(n_vocab=N_VOCAB)
# state_dict = torch.load("models/chess-transform2.pth")
# model.load_state_dict(state_dict)

# with open("data/move_encoder.pkl", "rb") as f:
#     encoder = pickle.load(f)


@torch.no_grad()
def finish_game(model, sequence=None, n_moves=30):
    if sequence is None:
        sequence = torch.tensor([1]).unsqueeze(0)
    for _ in range(n_moves):
        out = model(sequence)
        first = torch.tensor([1]).unsqueeze(0)
        sequence = torch.cat((first, out.argmax(dim=2)))
    return sequence.squeeze(0, 1).numpy()
