from chess_seq.tictactoe import mechanics
from chess_seq.tictactoe.players import Player
import random
import numpy as np


class TTTEnv:
    """
    Natural environment for playing Tic Tac Toe.
    Could be made faster by using only tokenids instead of going to tokens and back?

    states: sequence of moves np.array((game_length, 2)). e.g.
    actions: pairs of square i, j
    rewards: +1 win, 0 tie, -1 loss, -2 illegal move
    done: game over or illegal move

    """

    def __init__(
        self,
        adversary: Player,
        agent_start=None,
        illegal_cost=-2,
        greedy_adversary=True,
    ):
        self.adversary = adversary
        self.agent_start = agent_start
        self.illegal_cost = illegal_cost
        self.greedy_adversary = greedy_adversary
        self.reset()

    def reset(self, agent_start=None):
        if agent_start is None and self.agent_start is not None:
            agent_start = self.agent_start
        elif agent_start is None and self.agent_start is None:
            agent_start = random.choice([True, False])

        self.game = mechanics.TTTBoard()
        if agent_start:
            self.agent_identity = "X"
        else:
            self.agent_identity = "O"
            self._adversary_play()
        return self._get_state(), "Game started."

    def step(self, action):
        if action not in self.game.legal_moves:
            # print(f"Illegal move: {action}. Legal moves: {self.game.legal_moves}")
            return self._get_state(), self.illegal_cost, False, True, "Illegal move."

        self.game.push(action)
        if self.game.is_game_over():
            return self.get_game_over_state()

        self._adversary_play()
        if self.game.is_game_over():
            return self.get_game_over_state()

        return self._get_state(), 0, False, False, "Keep playing."

    def _adversary_play(self):
        adv_move = self.adversary.get_move(self.game, greedy=self.greedy_adversary)
        assert adv_move in self.game.legal_moves, {
            "msg": "Adversary returned Bad move.",
            "move": adv_move,
            "legal_moves": self.game.legal_moves,
        }
        self.game.push(adv_move)

    def get_game_over_state(self):
        assert self.game.is_game_over(), "Game is not over yet."
        if self.game.winner == "T":
            reward = 0
        elif self.game.winner == self.agent_identity:
            reward = 1
        else:
            reward = -1
        return self._get_state(), reward, True, False, "Game over."

    def _get_state(self):
        return np.array(self.game.move_stack)
