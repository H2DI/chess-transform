from chess_seq.tictactoe.players import (
    RandomPlayer,
    SimplePlayer,
    NNPlayer,
    ReasonablePlayer,
    PlayVPlay,
)
from chess_seq.tictactoe.environment import TTTEnv
import numpy as np

player1 = RandomPlayer()
player2 = ReasonablePlayer()
N = 1000

results = []
scores = []
for _ in range(N):
    vs = PlayVPlay(player1, player2)
    vs.play_game()
    if vs.game.winner == "X":
        results.append(1)
        reward = 1 - len(vs.game.move_stack) * 0.01
        scores.append(reward)
    elif vs.game.winner == "O":
        results.append(-1)
        reward = -1 + len(vs.game.move_stack) * 0.01
        scores.append(reward)
    if vs.game.winner == "T":
        results.append(0)
        scores.append(0)
print(f"After {N} games when Random starts:")
print(f"Random wins: {results.count(1) / N}")
print(f"Random losses: {results.count(-1) / N}")
print(f"Random ties: {results.count(0) / N}")
print(f"Score : {np.mean(scores)}")

results = []
for _ in range(N):
    vs = PlayVPlay(player2, player1)
    vs.play_game()
    results.append(vs.game.winner)
    if vs.game.winner == "X":
        results.append(-1)
        reward = -1 + len(vs.game.move_stack) * 0.01
        scores.append(reward)
    elif vs.game.winner == "O":
        results.append(1)
        reward = 1 - len(vs.game.move_stack) * 0.01
        scores.append(reward)
    if vs.game.winner == "T":
        results.append(0)
        scores.append(0)
print(f"After {N} games when Reasonable starts:")
print(f"Random wins: {results.count(1) / N}")
print(f"Random losses: {results.count(-1) / N}")
print(f"Random ties: {results.count(0) / N}")
print(f"Score : {np.mean(scores)}")
