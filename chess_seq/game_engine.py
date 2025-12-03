import numpy as np
import random
import chess
import torch
from torch import nn

from chess_seq.encoder import InvalidMove
from chess_seq.encoder import MoveEncoder


class ChessGameEngine:
    def __init__(self, model: nn.Module, encoder: MoveEncoder, device=None):
        self.model = model
        self.encoder = encoder
        self.device = device if device else next(model.parameters()).device

        self.start_token_id = self.encoder.start_token_id
        self.end_token_id = self.encoder.end_token_id

    @torch.no_grad()
    def generate_sequence(self, sequence=None, n_plies=30):
        """
        outputs a sequence of tokens, regardless of whether this gives a proper
        chess game
        """
        model = self.model
        device = self.device

        if sequence is None:
            sequence = torch.tensor(np.array([[self.start_token_id]]), device=device)
        for _ in range(n_plies):
            out = model(sequence)  # B, T, vocab_size
            next_id = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            sequence = torch.cat((sequence, next_id), dim=1)
            if sequence[0, -1].cpu().numpy() == self.end_token_id:
                break
        return sequence.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def play_game(self, game=None, n_plies=30, record_pgn=True, greedy=True):
        """
        Plays a chess game. A random move is chosen if the model's output is not a valid
        move.
        """

        device = self.device
        encoder = self.encoder
        if game is None:
            game = chess.Board()
            sequence = torch.tensor(np.array([[self.start_token_id]]), device=device)
        else:
            sequence = encoder.board_to_sequence(game, end=False)
            sequence = torch.tensor(np.array([sequence]), device=device)

        if record_pgn:
            pgn_game = (
                encoder.board_to_pgn(game) if game is not None else chess.pgn.Game()
            )
            node = pgn_game.end()
            del pgn_game.headers["Date"]
            del pgn_game.headers["Result"]
        else:
            pgn_game = None
            node = None

        current_ply = 1
        bad_plies = []

        for _ in range(n_plies):
            token_id, ended = self._generate_next_token_id(sequence, greedy=greedy)
            if ended:
                break
            current_ply += 1
            game, node, bad_plies, sequence = self._play_move_or_fallback(
                token_id,
                game,
                node,
                bad_plies,
                current_ply,
                sequence,
                record_pgn,
            )

            if game.is_game_over():
                break
        return game, pgn_game, bad_plies

    def _generate_next_token_id(self, sequence, greedy=True, no_end=True):
        out = self.model(sequence)  # B, T, vocab_size
        if greedy:
            next_token_id = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        else:
            logits = out[0, -1]
            if no_end:
                logits[0] = float("-inf")
            dist = torch.distributions.Categorical(logits=logits)
            next_token_id = dist.sample().unsqueeze(0).unsqueeze(0)
        return next_token_id, False

    def _play_move_or_fallback(
        self,
        token_id,
        game,
        node,
        bad_plies,
        current_ply,
        sequence,
        record_pgn,
    ):
        """
        Updates the game, pgn, node *and* the puts the tokens of the random move
        actually played in the sequence
        """
        game, node, played_proposed_move, true_move_played = self._play_from_token(
            token_id, game, node, record_pgn
        )

        if not played_proposed_move:
            bad_plies.append(current_ply)
            token_id = np.array([[self.encoder.move_to_id(true_move_played)]])

        sequence = torch.cat(
            (sequence, torch.tensor(token_id, device=self.device)),
            dim=1,
        )
        return game, node, bad_plies, sequence

    @torch.no_grad()
    def _play_from_token(self, token_id, game, node, record_pgn):
        """
        Takes the token and:
        - plays the move if it is valid
        - plays a random move otherwise
        Returns also a boolean saying if the move was legal
        """
        full_move = self.encoder.inverse_transform(token_id)
        full_move = "".join(full_move)
        try:
            chess_move = self.encoder.token_to_move(full_move)
            if chess_move in game.legal_moves:
                game.push(chess_move)
                if record_pgn and node is not None:
                    node = node.add_variation(chess_move)
                played_proposed_move = True
                return game, node, played_proposed_move, chess_move
            else:
                raise InvalidMove(f"{chess_move} is illegal")
        except InvalidMove as e:
            legal_moves = list(game.legal_moves)
            random_move = random.choice(legal_moves)
            game.push(random_move)
            if record_pgn and node is not None:
                node = node.add_variation(random_move)
                node.nags.add(chess.pgn.NAG_BLUNDER)
                node.comment = f"{e}. Played random move."
            played_proposed_move = False
            return game, node, played_proposed_move, random_move
