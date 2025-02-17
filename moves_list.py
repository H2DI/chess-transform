import pickle
from sklearn.preprocessing import LabelEncoder


def generate_all_moves():
    columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
    lines = [str(i) for i in range(1, 9)]
    all_moves = ["START", "END"]
    for column in columns:
        for line in lines:
            all_moves.append(column + line)
    return all_moves


move_encoder = LabelEncoder()
move_encoder.fit(generate_all_moves())
print(len(move_encoder.classes_))

with open("data/move_encoder.pkl", "wb") as f:
    pickle.dump(move_encoder, f)
