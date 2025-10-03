import torch
import chess

from chess_seq.training.trainer import log_stat_group
from chess_seq.evaluation.game_engine import ChessGameEngine


@torch.no_grad()
def print_basic_games(model, encoder):
    engine = ChessGameEngine(model, encoder)
    game_tokens = engine.finish_game(model, encoder, n_plies=44)
    print("\n")
    print(encoder.inverse_transform(game_tokens))

    print("\n Full game:")
    game, pgn, bad_plies = engine.play_game(n_plies=30)
    print(
        f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
        if bad_plies
        else "0 bad moves"
    )
    print(pgn.mainline_moves())

    print("\n Full game after 1.f3")
    game = chess.Board()
    game.push(chess.Move.from_uci("f2f3"))
    game, pgn, bad_plies = engine.play_game(game=game, n_plies=30)
    print(
        f"{len(bad_plies)} bad moves. First bad ply: {bad_plies[0]}"
        if bad_plies
        else "0 bad moves"
    )
    print(pgn.mainline_moves())


@torch.no_grad()
def test_first_moves(model, encoder, n_plies=30, prints=False):
    engine = ChessGameEngine(model, encoder)

    game = chess.Board()
    first_moves = list(game.legal_moves)
    all_number_of_bad_plies = []
    all_first_bad_plies = []
    for move in first_moves:
        game = chess.Board()
        game.push(move)
        game, _, bad_plies = engine.play_game(
            game=game, n_plies=n_plies, record_pgn=False
        )
        if prints:
            print(f"{move=}")
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
    engine = ChessGameEngine(model, encoder)
    sequence = engine.generate_sequence()
    print(encoder.inverse_transform(sequence))

    for n_plies in lengths:
        n_bad, t_first_bad = test_first_moves(model, encoder, n_plies=n_plies)
        log_stat_group(writer, f"Play{n_plies}/NumberOfBadMoves", n_bad, n_games)
        log_stat_group(writer, f"Play{n_plies}/FirstBadMoves", t_first_bad, n_games)
