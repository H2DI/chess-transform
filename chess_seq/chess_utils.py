import chess
import chess.pgn

import csv


def parse_move(move):
    """Convert a chess.Move object to a list of three tokens"""
    from_square = chess.square_name(move.from_square)
    to_square = chess.square_name(move.to_square)
    if move.promotion:
        promotion = "p" + chess.piece_symbol(move.promotion)
    else:
        promotion = "p" + "None"
    return [from_square, to_square, promotion]


def parse_board(board, encoder, end=True):
    """Convert a chess.Board object to a list of tokens"""
    moves_list = ["START"]
    for move in board.move_stack:
        moves_list += parse_move(move)
    if end:
        moves_list.append("END")
    return encoder.transform(moves_list)


def board_to_pgn(board):
    pgn_game = chess.pgn.Game()
    node = pgn_game
    for move in board.move_stack:
        node = node.add_main_variation(move)
    return pgn_game


def parse_pgn_game(pgn_game, encoder):
    """ """
    board = pgn_game.board()
    moves_list = ["START"]
    for move in pgn_game.mainline_moves():
        moves_list += parse_move(move)
        board.push(move)
    moves_list.append("END")
    return encoder.transform(moves_list)


def process_pgn_file(input_file, output_file, encoder, end_id=None, start_id=0):
    with open(input_file, "r") as pgn, open(output_file, "w", newline="") as out:
        csv_writer = csv.writer(out)
        csv_writer.writerow(["game_id", "moves"])

        game_id = start_id
        while True:
            if end_id is not None and game_id >= end_id:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            encoded_moves = parse_pgn_game(game, encoder)
            csv_writer.writerow(
                [game_id, " ".join([str(move) for move in encoded_moves])]
            )
            game_id += 1
