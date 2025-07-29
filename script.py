import chess
import pandas as pd
import numpy as np

import os
# import re

### convert to npz
# file_path = "data/train_csvs/lichess_elite_2014-10.csv"

# game = chess.Board()
# df = pd.read_csv(file_path, header=None, skiprows=1)
# df = df.iloc[:, 1:]  # drop first column

# print(df)

# print(df.values.shape)

# # np.savez("data/train_npz/lichess_elite_2013-09.npz", data=df.values)


# loaded = np.load("data/train_npz/lichess_elite_2013-09.npz", allow_pickle=True)
# data = loaded["data"]
# print(data.shape)

total_lines = 0

folder_path = "data/train_csvs"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename.lower() == ".ds_store":
        continue
    if os.path.isfile(file_path):
        print(f"Processing file: {file_path}")
    with open(file_path, "r") as f:
        num_lines = sum(1 for _ in f)
    total_lines += num_lines
    # print(f"Number of games in training CSV: {num_lines}")

print(total_lines)
