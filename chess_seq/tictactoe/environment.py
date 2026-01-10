from chess_seq.tictactoe import mechanics
from chess_seq.tictactoe.players import Player
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
    ):
        self.adversary = adversary
        self.illegal_cost = illegal_cost
        self.agent_start = agent_start
        self.set_new_prompt()
        self.reset_to_prompt()

    def reset(self, agent_start=None):
        if agent_start is None and self.agent_start is None:
            agent_start = np.random.rand() < 0.5
        elif agent_start is None:
            agent_start = self.agent_start

        self.game = mechanics.TTTBoard()
        if agent_start:
            self.agent_id = "X"
        else:
            self.agent_id = "O"
            self._adversary_play()

        return self._get_state(), {
            "msg": "Game started.",
            "agent_id": self.agent_id,
            "legal_moves": self.game.legal_moves,
        }

    def set_new_prompt(self):
        if self.agent_start is None:
            prompt_agent_start = np.random.rand() < 0.5
        else:
            prompt_agent_start = self.agent_start

        if prompt_agent_start:
            self.prompt_id = "X"
            self.prompt = []
        else:
            self.prompt_id = "O"
            game = mechanics.TTTBoard()
            adv_move = self.adversary.get_move(game)
            self.prompt = [adv_move]

    def reset_to_prompt(self):
        self.game = mechanics.TTTBoard()
        self.agent_id = self.prompt_id
        for action in self.prompt:
            self.game.push(action)

        return self._get_state(), {
            "msg": "Game started.",
            "agent_id": self.agent_id,
            "legal_moves": self.game.legal_moves,
        }

    def step(self, action):
        if action not in self.game.legal_moves:
            return (
                self._get_state(),
                self.illegal_cost,
                False,
                True,
                {"msg": "Illegal move."},
            )

        self.game.push(action)
        if self.game.is_game_over():
            return self.get_game_over_state()

        self._adversary_play()
        if self.game.is_game_over():
            return self.get_game_over_state()

        return (
            self._get_state(),
            0,
            False,
            False,
            {"msg": "Keep playing.", "legal_moves": self.game.legal_moves},
        )

    def _adversary_play(self):
        adv_move = self.adversary.get_move(self.game)
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
        elif self.game.winner == self.agent_id:
            reward = 1 - len(self.game.move_stack) * 0.01
        else:
            reward = -1 + len(self.game.move_stack) * 0.01
        return (
            self._get_state(),
            reward,
            True,
            False,
            {"msg": "Game over.", "winner": self.game.winner},
        )

    def _get_state(self):
        return np.array(self.game.move_stack)
