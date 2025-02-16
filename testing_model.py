import torch

import chess_transformer
import pickle

import datasets
import matplotlib.pyplot as plt
import numpy as np

N_VOCAB = 36727

# model = chess_transformer.ChessNet2(n_vocab=N_VOCAB)
# state_dict = torch.load("models/chess-transform2.pth")
# model.load_state_dict(state_dict)

# with open("data/move_encoder.pkl", "rb") as f:
#     encoder = pickle.load(f)


@torch.no_grad()
def finish_game(model, encoder, sequence=[""], n_moves=10, encoded=False):
    if not (encoded):
        encoded_seq = torch.tensor(encoder.transform(sequence)).unsqueeze(0)
    else:
        encoded_seq = torch.tensor(sequence).unsqueeze(0)
    for _ in range(n_moves):
        next_move = model(encoded_seq).argmax(dim=1).unsqueeze(0)
        encoded_seq = torch.cat((encoded_seq, next_move), dim=1)
    game = encoded_seq.squeeze(0, 1).numpy()
    return encoder.inverse_transform(game)


# first = ["", "c4"]
# finished_game = finish_game(model, encoder, first)
# print(finished_game)
# first = ["", "d4"]
# finished_game = finish_game(model, encoder, first)
# print(finished_game)
# first = [""]
# finished_game = finish_game(model, encoder, first)
# print(finished_game)

# learned_game = datasets.generate_games_list(first_game=0, n_games=1)[0]
# correct_moves = []
# for i in range(1, len(learned_game)):
#     predicted_move = finish_game(
#         model, encoder, sequence=learned_game[:i], n_moves=1, encoded=True
#     )[-1]
#     true_move = encoder.inverse_transform([learned_game[i]])[0]
#     print("Predicted: ", predicted_move, " True: ", true_move)
#     correct_moves.append(true_move == predicted_move)

# print(np.mean(correct_moves))

# plt.show()
