import pickle
from sklearn.preprocessing import LabelEncoder

from chess_seq.tictactoe.mechanics import tokens_list

move_encoder = LabelEncoder()
move_encoder.fit(tokens_list())
print(len(move_encoder.classes_))

with open("data/ttt_encoder.pkl", "wb") as f:
    pickle.dump(move_encoder, f)
