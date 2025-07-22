import numpy as np
import torch
import random
import chess

from chess_seq.chess_utils.chess_utils import (
    move_to_tokens,
    board_to_sequence,
    board_to_pgn,
    tokens_to_move,
    InvalidMove,
)


class ChessGameEngine:
    def __init__(self, model, encoder, device=None):
        self.model = model
        self.encoder = encoder
        self.device = device if device else next(model.parameters()).device

        self.end_token = np.array(encoder.transform(["END"]))

    @torch.no_grad()
    def generate_sequence(self, sequence=None, n_plies=30):
        """
        outputs a sequence of tokens, regardless of whether this gives a proper
        chess game
        """
        model = self.model
        encoder = self.encoder
        device = self.device

        if sequence is None:
            sequence = torch.tensor(
                np.array([encoder.transform(["START"])]), device=device
            )
        for _ in range(3 * n_plies):
            out = model(sequence)  # B, T, vocab_size
            next_move = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            sequence = torch.cat((sequence, next_move), dim=1)
            if sequence[0, -1].cpu().numpy() == self.end_token:
                break
        return sequence.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def play_game(self, game=None, n_plies=30, record_pgn=True):
        """
        Plays a chess game. A random move is chosen if the model's output is not a valid
        move.
        """

        device = self.device
        encoder = self.encoder
        if game is None:
            game = chess.Board()
            sequence = torch.tensor(
                np.array([encoder.transform(["START"])]), device=device
            )
        else:
            sequence = board_to_sequence(game, encoder, end=False)
            sequence = torch.tensor(np.array([sequence]), device=device)

        if record_pgn:
            pgn_game = board_to_pgn(game) if game is not None else chess.pgn.Game()
            node = pgn_game.end()
            del pgn_game.headers["Date"]
            del pgn_game.headers["Result"]
        else:
            pgn_game = None
            node = None

        current_ply = 1
        bad_plies = []

        for _ in range(n_plies):
            tokens, candidate_sequence, ended = self._generate_next_tokens(sequence)
            if ended:
                break
            current_ply += 1
            game, node, bad_plies, sequence = self._play_move_or_fallback(
                tokens,
                game,
                node,
                bad_plies,
                current_ply,
                sequence,
                candidate_sequence,
                record_pgn,
            )

            if game.is_game_over():
                break
        return game, pgn_game, bad_plies

    def _generate_next_tokens(self, sequence):
        tokens = []
        candidate_sequence = sequence.clone()
        for _ in range(3):
            out = self.model(candidate_sequence)  # B, T, vocab_size
            next_token = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            tokens.append(next_token.item())
            candidate_sequence = torch.cat((candidate_sequence, next_token), dim=1)
            if candidate_sequence[0, -1].cpu().numpy() == self.end_token:
                return tokens, candidate_sequence, True
        return tokens, candidate_sequence, False

    def _play_move_or_fallback(
        self,
        tokens,
        game,
        node,
        bad_plies,
        current_ply,
        sequence,
        candidate_sequence,
        record_pgn,
    ):
        """
        Updates the game, pgn, node *and* the puts the tokens of the random move
        actually played in the sequence
        """
        game, node, legal, true_move_played = self._play_from_tokens(
            tokens, game, node, record_pgn
        )

        if not legal:
            bad_plies.append(current_ply)
            tokens = np.array(
                [self.encoder.transform(move_to_tokens(true_move_played))]
            )
            sequence = torch.cat(
                (sequence, torch.tensor(tokens, device=self.device)),
                dim=1,
            )
        else:
            sequence = candidate_sequence
        return game, node, bad_plies, sequence

    @torch.no_grad()
    def _play_from_tokens(self, tokens, game, node, record_pgn):
        """
        Takes the three tokens and:
        - plays the move if it is valid
        - plays a random move otherwise
        Returns also a boolean saying if the move was legal
        """
        full_move = self.encoder.inverse_transform(
            tokens
        )  # list of 3 strings (from, to, promo)
        full_move = "".join(full_move)
        try:
            chess_move = tokens_to_move(full_move)
            if chess_move in game.legal_moves:
                game.push(chess_move)
                if record_pgn and node is not None:
                    node = node.add_variation(chess_move)
                return game, node, True, chess_move
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
            return game, node, False, random_move
