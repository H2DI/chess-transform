import random
import torch

import chess_seq.utils as utils
from chess_seq.tictactoe import mechanics
from chess_seq.tictactoe.game_engine import TTTGameEngine
from chess_seq.tictactoe.mechanics import TTTBoard
from torch import no_grad


class TTTAgent:
    def __init__(self, model, encoder, device=None, full_name="Generic_TTTAgent"):
        self.engine = TTTGameEngine(model, encoder, device=device)
        self.full_name = full_name
        self.last_tokenid = None

    def new_game(self):
        pass

    @no_grad()
    def get_action(self, game_moves, greedy=True):
        """ """
        self.engine.model.eval()
        tokenid = self._get_tokenid(game_moves, greedy=greedy)
        token = self.engine.encoder.inverse_transform(tokenid[0])[0]
        move = mechanics.tokens_to_move(token, corrected=True)
        return move

    def _get_tokenid(self, game_moves, greedy=True):
        sequence = mechanics.move_stack_to_sequence(
            game_moves, self.engine.encoder, self.engine.device
        )
        tokenid, _ = self.engine.generate_next_tokenid(sequence, greedy=greedy)
        self.last_tokenid = tokenid
        return tokenid

    def load_checkpoint(self, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_path = utils.get_latest_checkpoint(self.full_name)
        else:
            checkpoint_path = f"checkpoints/{self.full_name}/{checkpoint_name}.pth"

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.engine.device,
            weights_only=False,
        )
        self.engine.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def save_checkpoint(self, model_config, checkpoint_name=None):
        pass


class RandomAgent:
    def __init__(self):
        self._reset_game()

    def _reset_game(self):
        self.internal_game = TTTBoard()

    def new_game(self):
        self._reset_game()

    def get_action(self, game_moves):
        if len(game_moves) == 0:
            return self._push_random_move()
        (i, j) = game_moves[-1]
        self.internal_game.push((int(i), int(j)))

        assert not self.internal_game.is_game_over(), "Game is already over."
        return self._push_random_move()

    def _push_random_move(self):
        move = random.choice(self.internal_game.legal_moves)
        self.internal_game.push(move)
        return move
