import torch

import numpy as np
import chess
import chess.pgn
import random

from .chess_utils import parse_move, parse_board, board_to_pgn
from .training import log_stat_group


class InvalidMove(Exception):
    """Exception raised for invalid chess moves."""

    pass


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


@torch.no_grad()
def finish_game(model, encoder, sequence=None, n_plies=30):
    """
    outputs a sequence of tokens, regardless of whether this gives a proper
    chess game
    """
    device = next(model.parameters()).device
    end_token = np.array(encoder.transform(["END"]))

    if sequence is None:
        sequence = torch.tensor(np.array([encoder.transform(["START"])]), device=device)
    for _ in range(3 * n_plies):
        out = model(sequence)  # B, T, vocab_size
        next_move = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        sequence = torch.cat((sequence, next_move), dim=1)
        if sequence[0, -1].cpu().numpy() == end_token:
            break
    return sequence.squeeze(0).cpu().numpy()


@torch.no_grad()
def play_valid_game(model, encoder, game=None, n_plies=30):
    """
    Plays a chess game. A random move is chosen if the model's output is not a valid
    move.
    """
    device = next(model.parameters()).device
    end_token = np.array(encoder.transform(["END"]))
    if game is None:
        game = chess.Board()
        pgn_game = chess.pgn.Game()
        sequence = torch.tensor(np.array([encoder.transform(["START"])]), device=device)
    else:
        pgn_game = board_to_pgn(game)
        sequence = parse_board(game, encoder, end=False)
        sequence = torch.tensor(np.array([sequence]), device=device)

    node = pgn_game.end()
    current_ply = 1
    bad_plies = []
    # Remove empty fields in the PGN header
    del pgn_game.headers["Date"]
    del pgn_game.headers["Result"]

    for _ in range(n_plies):
        tokens = []
        candidate_sequence = sequence.clone()
        # compute next ply/three tokens
        for _ in range(3):
            out = model(candidate_sequence)  # B, T, vocab_size
            next_token = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            tokens.append(next_token.item())
            candidate_sequence = torch.cat((candidate_sequence, next_token), dim=1)
            if candidate_sequence[0, -1].cpu().numpy() == end_token:
                break

        game, node, legal, true_move_played = play_from_tokens(
            tokens, game, node, encoder
        )

        if not legal:
            bad_plies.append(current_ply)
            tokens = np.array([encoder.transform(parse_move(true_move_played))])
            sequence = torch.cat(
                (sequence, torch.tensor(tokens, device=device)),
                dim=1,
            )
        else:
            sequence = candidate_sequence

        current_ply += 1

        if game.is_game_over():
            break
    return game, pgn_game, bad_plies


@torch.no_grad()
def play_from_tokens(tokens, game, node, encoder):
    full_move = encoder.inverse_transform(tokens)  # list of 3 strings (from, to, promo)
    full_move = "".join(full_move)
    try:
        chess_move = tokens_to_move(full_move)
        if chess_move in game.legal_moves:
            game.push(chess_move)
            node = node.add_variation(chess_move)
            return game, node, True, chess_move
        else:
            raise InvalidMove(f"{chess_move} is illegal")
    except InvalidMove as e:
        legal_moves = list(game.legal_moves)
        random_move = random.choice(legal_moves)
        game.push(random_move)
        node = node.add_variation(random_move)
        node.nags.add(chess.pgn.NAG_BLUNDER)
        node.comment = f"{e}. Played random move."
        return game, node, False, random_move


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


def eval_legal_moves_and_log(model, encoder, writer, n_games):
    model.eval()
    print_basic_games(model, encoder)

    n_plies = 30
    n_bad, t_first_bad = test_first_moves(model, encoder, n_plies=30)
    log_stat_group(writer, "Play/NumberOfBadMoves", n_bad, n_games)
    log_stat_group(writer, "Play/FirstBadMoves", t_first_bad, n_games)

    n_plies = 200
    n_bad, t_first_bad = test_first_moves(model, encoder, n_plies=n_plies)
    log_stat_group(writer, f"Play{n_plies}/NumberOfBadMoves", n_bad, n_games)
    log_stat_group(writer, f"Play{n_plies}/FirstBadMoves", t_first_bad, n_games)
