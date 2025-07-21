import chess
import chess.pgn


class InvalidMove(Exception):
    """Exception raised for invalid chess moves."""

    pass


def move_to_tokens(move):
    """Convert a chess.Move object to a list of three tokens"""
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    if move.promotion:
        promotion = "p" + chess.piece_symbol(move.promotion)
    else:
        promotion = "p" + "None"
    return [from_square, to_square, promotion]


def board_to_sequence(board, encoder, end=True):
    """Convert a chess.Board object to a list of tokens"""
    moves_list = ["START"]
    for move in board.move_stack:
        moves_list += move_to_tokens(move)
    if end:
        moves_list.append("END")
    return encoder.transform(moves_list)


def board_to_pgn(board):
    pgn_game = chess.pgn.Game()
    node = pgn_game
    for move in board.move_stack:
        node = node.add_main_variation(move)
    return pgn_game


def pgn_to_sequence(pgn_game, encoder):
    """ """
    board = pgn_game.board()
    moves_list = ["START"]
    for move in pgn_game.mainline_moves():
        moves_list += move_to_tokens(move)
        board.push(move)
    moves_list.append("END")
    return encoder.transform(moves_list)


def tokens_to_move(move):
    """Convert a move from the encoder format to chess.Move."""
    if move == "START":
        return None
    elif move == "END":
        return None
    else:
        from_string, to_string, promo_string = move[:2], move[2:4], move[4:]
        if from_string[0] == "p" or to_string[0] == "p" or promo_string[0] != "p":
            raise InvalidMove(f"Invalid move format: {move}")

        from_square = chess.parse_square(from_string)
        to_square = chess.parse_square(to_string)
        if promo_string == "pNone":
            promotion = None
        else:
            promotion = chess.Piece.from_symbol(promo_string[1:])

        move_obj = chess.Move(
            from_square,
            to_square,
            promotion=promotion.piece_type if promotion else None,
        )
        return move_obj
