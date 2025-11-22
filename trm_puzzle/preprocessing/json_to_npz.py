import json
import numpy as np
import chess
import argparse

from chess_seq.encoder import MoveEncoder

# {
#     "PuzzleId": "001cr",
#     "FEN": "8/3B2pp/p5k1/2p3P1/1p1p1K2/8/1P6/8 b - - 0 38",
#     "Moves": "c5c4 d7e8"
#   },

# ---------------------------------------------------------
#  Mapping from piece to token ID (0–12)
# ---------------------------------------------------------


class BoardEncoder:
    def __init__(self):
        self.piece_to_id = {
            None: 0,
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6,
        }

    def piece_token(self, piece):
        if piece is None:
            return 0
        base = self.piece_to_id[piece.piece_type]
        return base if piece.color == chess.WHITE else base + 6

    def encode_position(self, board: chess.Board):
        tokens = []

        # 1) 64 square tokens
        for square in chess.SQUARES:
            tokens.append(self.piece_token(board.piece_at(square)))

        # 2) Castling (4 tokens)
        tokens.append(13 if board.has_kingside_castling_rights(chess.WHITE) else 14)
        tokens.append(15 if board.has_queenside_castling_rights(chess.WHITE) else 16)
        tokens.append(17 if board.has_kingside_castling_rights(chess.BLACK) else 18)
        tokens.append(19 if board.has_queenside_castling_rights(chess.BLACK) else 20)

        # 3) Turn (1 token)
        tokens.append(21 if board.turn == chess.WHITE else 22)

        return np.array(tokens, dtype=np.int16)  # shape (69,)


# ---------------------------------------------------------
#  Convert one puzzle example
# ---------------------------------------------------------


def encode_example(board_encoder, move_encoder, fen, opp_uci, sol_uci):
    board = chess.Board(fen)

    # Apply opponent move
    opp = board.parse_uci(opp_uci)
    board.push(opp)

    # Encode x and y
    x = board_encoder.encode_position(board)
    y = move_encoder.move_to_id(board.parse_uci(sol_uci))
    return x, y


# ---------------------------------------------------------
#  Main conversion script
# ---------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input json file")
    parser.add_argument("--output", required=True, help="output .npz file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    move_encoder = MoveEncoder()
    move_encoder.load("data/move_encoder.pkl")
    board_encoder = BoardEncoder()

    ids = []
    X = []
    Y = []

    for entry in data:
        puzzle_id = entry["PuzzleId"]
        fen = entry["FEN"]
        moves = entry["Moves"].split()

        if len(moves) < 2:
            continue  # skip malformed puzzles

        opp_move = moves[0]
        solution = moves[1]

        x, y = encode_example(board_encoder, move_encoder, fen, opp_move, solution)

        ids.append(puzzle_id)
        X.append(x)
        Y.append(y)

    ids = np.array(ids)
    X = np.stack(X)
    Y = np.stack(Y)

    np.savez_compressed(args.output, ids=ids, X=X, Y=Y)
    print(f"Saved {len(ids)} puzzles to {args.output}")


if __name__ == "__main__":
    main()
