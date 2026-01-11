import torch
import chess

from ..game_engine import ChessGameEngine
from ..datasets.preprocessing import build_dataloader


npz_path = None


data_loader = build_dataloader(
    npz_path,
    batch_size=16,
    padding_value=4610,
    max_length=500,
    shuffle=False,
)

def perform_eval(model, data_loader):


# todo:

# 6000 games
# compute perplexity
# compute top 1, top 3, top 5, top 10
# separate black and white
# plot top 5 accuracy per ply bucket


# compute illegal moves in top1 in next 10 plies after 11, 22, 33 plies
