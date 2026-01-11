import random
import torch

import chess_seq.utils.save_and_load as save_and_load
from chess_seq.tictactoe import mechanics
from chess_seq.tictactoe.game_engine import TTTGameEngine
from chess_seq.tictactoe.mechanics import TTTBoard
from torch import no_grad
from config import ModelConfig


class TTTAgent:
    def __init__(
        self,
        model_config: ModelConfig,
        model,
        device=None,
        full_name="Generic_TTTAgent",
    ):
        self.engine = TTTGameEngine(model_config, model, device=device)
        self.full_name = full_name
        self.last_tokenid = None
        self.last_token_entropy = None
        self.temperature = 0.0

    def new_game(self, agent_id):
        self.agent_id = agent_id  # "X" or "O"

    def _build_mask(self, legal_moves):
        mask = torch.zeros(self.engine.vocab_size, device=self.engine.device)
        mask = mask + float("-inf")
        for move in legal_moves:
            token = mechanics.move_to_tokens(move)
            tokenid = self.engine.encoder.transform([token])[0]
            mask[tokenid] = 0.0
        return mask

    @no_grad()
    def get_action(self, game_moves, temperature=None, legals=None):
        """ """
        if temperature is None:
            temperature = self.temperature
        self.engine.model.eval()
        mask = self._build_mask(legals) if legals is not None else None
        tokenid, entropy = self._get_tokenid(
            game_moves, temperature=temperature, mask=mask
        )
        self.last_token_entropy = entropy
        token = self.engine.encoder.inverse_transform(tokenid[0])[0]
        move = mechanics.tokens_to_move(token, corrected=True)
        return move

    def _get_tokenid(self, game_moves, temperature=0.0, mask=None):
        sequence = mechanics.move_stack_to_sequence(
            game_moves, self.engine.encoder, self.engine.device
        )
        tokenid, _, entropy = self.engine.generate_next_tokenid(
            sequence, temperature=temperature, mask=mask
        )
        self.last_tokenid = tokenid
        return tokenid, entropy

    def load_checkpoint(self, checkpoint_name=None):
        if checkpoint_name is None:
            checkpoint_path = save_and_load.get_latest_checkpoint(self.full_name)
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

    def new_game(self, agent_id):
        self.agent_id = agent_id  # "X" or "O"
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
