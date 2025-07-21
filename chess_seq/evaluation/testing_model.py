import torch

import numpy as np
import chess
import chess.pgn
import random

from chess_seq.chess.chess_utils import move_to_tokens, board_to_sequence, board_to_pgn
from chess_seq.training.trainer import log_stat_group


@torch.no_grad()
def print_basic_games(model, encoder):
    game_tokens = finish_game(model, encoder, n_plies=44)
    print("\n")
    print(encoder.inverse_transform(game_tokens))

    print("\n Full game:")
    game, pgn, bad_plies = play_valid_game(model, encoder, n_plies=30)
    print(
        f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
        if bad_plies
        else "0 bad moves"
    )
    print(pgn.mainline_moves())

    print("\n Full game after 1.f3")
    game = chess.Board()
    game.push(chess.Move.from_uci("f2f3"))
    game, pgn, bad_plies = play_valid_game(model, encoder, game=game, n_plies=30)
    print(
        f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
        if bad_plies
        else "0 bad moves"
    )
    print(pgn.mainline_moves())


@torch.no_grad()
def test_first_moves(model, encoder, n_plies=30, prints=False):
    game = chess.Board()
    first_moves = list(game.legal_moves)
    all_number_of_bad_plies = []
    all_first_bad_plies = []
    for move in first_moves:
        game = chess.Board()
        game.push(move)
        game, pgn, bad_plies = play_valid_game(
            model, encoder, game=game, n_plies=n_plies
        )
        if prints:
            print("")
            print(move)
            print(
                f"{len(bad_plies)} bad moves out of {len(game.move_stack)}. "
                + f"First bad ply: {bad_plies[0]}"
                if bad_plies
                else "0 bad moves"
            )
        all_number_of_bad_plies.append(len(bad_plies))
        all_first_bad_plies.append(bad_plies[0] if bad_plies else len(game.move_stack))
    return all_number_of_bad_plies, all_first_bad_plies


def eval_legal_moves_and_log(model, encoder, writer, n_games, lengths):
    model.eval()
    print_basic_games(model, encoder)

    for n_plies in lengths:
        n_bad, t_first_bad = test_first_moves(model, encoder, n_plies=n_plies)
        log_stat_group(writer, f"Play{n_plies}/NumberOfBadMoves", n_bad, n_games)
        log_stat_group(writer, f"Play{n_plies}/FirstBadMoves", t_first_bad, n_games)
