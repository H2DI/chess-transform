import chess
import chess.pgn

from itertools import product

import json
from pathlib import Path

import numpy as np


class InvalidMove(Exception):
    """Exception raised for invalid chess moves."""

    pass


class MoveEncoder:
    def __init__(self):
        super().__init__()
        self.start_token = "<start>"
        self.end_token = "<end>"

        self.id_to_token = []
        self.token_to_id = {}

    def build(self):
        self.id_to_token = self._generate_all_tokens()
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

        self.start_token_id = self.token_to_id[self.start_token]
        self.end_token_id = self.token_to_id[self.end_token]
        print("MoveEncoder built with vocabulary size:", len(self.id_to_token))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.id_to_token, f, ensure_ascii=False, separators=(",", ":"))
        print(f"Saved MoveEncoder id_to_token to {path}")

    def load(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            self.id_to_token = json.load(f)
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.start_token_id = self.token_to_id[self.start_token]
        self.end_token_id = self.token_to_id[self.end_token]
        print(f"Loaded MoveEncoder from {path}")
        return self

    def transform(self, tokens):
        return np.array([self.token_to_id[t] for t in tokens], dtype=np.int32)

    def inverse_transform(self, ids):
        return [self.id_to_token[i] for i in ids]

    def _generate_all_tokens(self):
        columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
        lines = [str(i) for i in range(1, 9)]
        squares = [c + ell for c, ell in product(columns, lines)]
        moves = [fr + to for fr, to in product(squares, repeat=2)]
        pieces = ["b", "n", "r", "q"]

        promotions = []
        for move in moves:
            if (move[1] == "7" and move[3] == "8") or (
                move[1] == "2" and move[3] == "1"
            ):
                promotions += [f"{move}p{piece}" for piece in pieces]

        return [self.start_token, self.end_token] + moves + promotions

    def move_to_id(self, move):
        return self.token_to_id[self.move_to_token(move)]

    def id_to_move(self, id: np.int32):
        return self.id_to_token[self.token_to_move(id)]

    def move_to_token(self, move):
        """Convert a chess.Move object to a token"""
        from_square = chess.square_name(move.from_square)
        to_square = chess.square_name(move.to_square)
        if move.promotion:
            promotion = "p" + chess.piece_symbol(move.promotion)
        else:
            promotion = ""
        return from_square + to_square + promotion

    def board_to_sequence(self, board, end=True):
        """Convert a chess.Board object to a list of tokens"""
        sequence = [self.start_token_id]
        for move in board.move_stack:
            sequence.append(self.move_to_id(move))
        if end:
            sequence.append(self.end_token_id)
        return sequence

    def board_to_pgn(self, board):
        pgn_game = chess.pgn.Game()
        node = pgn_game
        for move in board.move_stack:
            node = node.add_main_variation(move)
        return pgn_game

    def pgn_to_sequence(self, pgn_game):
        """ """
        board = pgn_game.board()
        sequence = [self.start_token_id]
        for move in pgn_game.mainline_moves():
            sequence.append(self.move_to_id(move))
            board.push(move)
        sequence.append(self.end_token_id)
        return sequence

    def token_to_move(self, token):
        """Convert a move from the encoder format to chess.Move."""
        if token == self.start_token:
            # return "Start"
            return None
        elif token == self.end_token:
            # return "End"
            return None

        else:
            from_string, to_string, promo_string = token[:2], token[2:4], token[4:]
            from_square = chess.parse_square(from_string)
            to_square = chess.parse_square(to_string)
            if promo_string == "":
                promotion = None
            else:
                promotion = chess.Piece.from_symbol(promo_string[1:])

            move_obj = chess.Move(
                from_square,
                to_square,
                promotion=promotion.piece_type if promotion else None,
            )
            return move_obj


if __name__ == "__main__":
    move_encoder = MoveEncoder()
    move_encoder.build()
    move_encoder.save("data/id_to_token.json")
    move_encoder.load("data/id_to_token.json")

    print("Number of tokens:", len(move_encoder.id_to_token))
    print(move_encoder.token_to_id["e2e4"])
    print(move_encoder.token_to_id["e7e8pq"])
    print(move_encoder.id_to_token[4405])
