import chess
import chess.pgn
import pickle

from sklearn.preprocessing import LabelEncoder


def can_move(piece_type, start, end):
    board = chess.Board(None)
    start_square = chess.parse_square(start)
    end_square = chess.parse_square(end)
    piece = chess.Piece.from_symbol(piece_type)
    board.set_piece_at(start_square, piece)

    return chess.Move(start_square, end_square) in list(board.legal_moves)


def generate_all_pawn_moves():
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    white_pawn_moves = []
    black_pawn_moves = []

    # White Pawn Moves: Forward Moves and Promotions
    for file in files:
        for rank in ["2", "3", "4", "5", "6"]:
            white_pawn_moves.append(file + str(int(rank) + 1))  # Single move
        white_pawn_moves.append(file + "4")  # Double move from rank 2

    # Black Pawn Moves: Forward Moves and Promotions
    for file in files:
        for rank in ["7", "6", "5", "4", "3"]:
            black_pawn_moves.append(file + str(int(rank) - 1))  # Single move
        black_pawn_moves.append(file + "5")  # Double move from rank 7

    # Promotions for White Pawns
    promotions = ["Q", "R", "B", "N"]
    for file in files:
        for promotion in promotions:
            white_pawn_moves.append(file + "8=" + promotion)

    # Promotions for Black Pawns
    for file in files:
        for promotion in promotions:
            black_pawn_moves.append(file + "1=" + promotion)

    # Captures for White Pawns
    for file in files:
        for rank in ["2", "3", "4", "5", "6", "7"]:
            if file != "a":
                white_pawn_moves.append(
                    file + "x" + chr(ord(file) - 1) + str(int(rank) + 1)
                )  # Capture left
            if file != "h":
                white_pawn_moves.append(
                    file + "x" + chr(ord(file) + 1) + str(int(rank) + 1)
                )  # Capture right

    # Captures for Black Pawns
    for file in files:
        for rank in ["7", "6", "5", "4", "3", "2"]:
            if file != "a":
                black_pawn_moves.append(
                    file + "x" + chr(ord(file) - 1) + str(int(rank) - 1)
                )  # Capture left
            if file != "h":
                black_pawn_moves.append(
                    file + "x" + chr(ord(file) + 1) + str(int(rank) - 1)
                )  # Capture right

    # Capture promotions for White Pawns
    for file in files:
        if file != "a":
            for promotion in promotions:
                white_pawn_moves.append(
                    file + "x" + chr(ord(file) - 1) + "8=" + promotion
                )
        if file != "h":
            for promotion in promotions:
                white_pawn_moves.append(
                    file + "x" + chr(ord(file) + 1) + "8=" + promotion
                )

    # Capture promotions for Black Pawns
    for file in files:
        if file != "a":
            for promotion in promotions:
                black_pawn_moves.append(
                    file + "x" + chr(ord(file) - 1) + "1=" + promotion
                )
                # black_pawn_moves.append(file + 'x' + chr(ord(file) - 1) + '1' + promotion.lower())
        if file != "h":
            for promotion in promotions:
                black_pawn_moves.append(
                    file + "x" + chr(ord(file) + 1) + "1=" + promotion
                )

    return white_pawn_moves + black_pawn_moves


def generate_all_piece_moves():
    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["1", "2", "3", "4", "5", "6", "7", "8"]
    pieces = ["N", "B", "R", "Q", "K"]
    piece_moves = []

    # Generate all possible moves for each piece type
    for piece in pieces:
        for start_file in files:
            for start_rank in ranks:
                for end_file in files:
                    for end_rank in ranks:
                        if not (
                            can_move(
                                piece, start_file + start_rank, end_file + end_rank
                            )
                        ):
                            continue

                        # Basic move
                        piece_moves.append(piece + end_file + end_rank)

                        # Capture
                        piece_moves.append(piece + "x" + end_file + end_rank)

                        if not (piece == "K"):
                            # File disambiguation
                            piece_moves.append(piece + start_file + end_file + end_rank)
                            piece_moves.append(
                                piece + start_file + "x" + end_file + end_rank
                            )
                            # Rank disambiguation
                            piece_moves.append(piece + start_rank + end_file + end_rank)
                            piece_moves.append(
                                piece + start_rank + "x" + end_file + end_rank
                            )

                        # Full disambiguation
                        if piece in ["N", "B", "Q"]:
                            piece_moves.append(
                                piece + start_file + start_rank + end_file + end_rank
                            )
                            piece_moves.append(
                                piece
                                + start_file
                                + start_rank
                                + "x"
                                + end_file
                                + end_rank
                            )
    return piece_moves


def generate_castling_moves():
    return ["O-O", "O-O-O"]


def add_checks_and_checkmates(moves):
    checked_moves = []
    for move in moves:
        checked_moves.append(move + "+")  # Check
        checked_moves.append(move + "#")  # Checkmate
    return moves + checked_moves


def generate_all_moves():
    # Generate all possible moves
    pawn_moves = generate_all_pawn_moves()
    piece_moves = generate_all_piece_moves()
    castling_moves = generate_castling_moves()

    # Combine and add checks/checkmates
    all_moves = pawn_moves + piece_moves + castling_moves
    all_moves = add_checks_and_checkmates(all_moves)

    # Remove duplicates
    return list(set(all_moves))


move_encoder = LabelEncoder()
move_encoder.fit(generate_all_moves())

with open("data/move_encoder.pkl", "wb") as f:
    pickle.dump(move_encoder, f)
