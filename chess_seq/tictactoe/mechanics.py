import numpy as np

# import random
import torch


# Move : (0, 1)  Tuple[int, int]
# Tokens: "{0},{1}" or "START" or "X" or "O" or "T" or "END"
# TokenID: int in [0, 12]


class InvalidMove(Exception):
    """Exception raised for invalid moves in TicTacToe."""


class TTTBoard:
    def __init__(self):
        self.X = np.zeros((3, 3))
        self.O = np.zeros((3, 3))
        self.turn = 0  #

        self.move_stack = []

        self._is_game_over = False
        self.winner = None
        self._update_legal_moves()

    def is_game_over(self):
        return self._is_game_over

    def copy(self):
        new_board = TTTBoard()
        new_board.X = self.X.copy()
        new_board.O = self.O.copy()
        new_board.turn = self.turn
        new_board.move_stack = self.move_stack.copy()
        new_board._is_game_over = self._is_game_over
        new_board.winner = self.winner
        new_board.legal_moves = self.legal_moves.copy()
        return new_board

    def push(self, x):
        assert not self.is_game_over(), "Game is already over."
        i, j = x[0], x[1]

        if (self.turn % 2) == 0:
            self.X[i, j] = 1
        else:
            self.O[i, j] = 1

        self.turn += 1
        self.move_stack.append(x)

        self._check_winner()
        self._update_legal_moves()

    def _check_winner(self):
        last_move = self.move_stack[-1]
        x, y = last_move[0], last_move[1]

        to_check = self.X if (self.turn % 2) == 1 else self.O

        if (
            to_check[x, :].sum() == 3
            or to_check[:, y].sum() == 3
            or to_check.diagonal().sum() == 3
            or to_check[::-1].diagonal().sum() == 3
        ):
            self._is_game_over = True
            self.winner = "X" if (self.turn % 2) == 1 else "O"

        if self.turn == 9 and not self.is_game_over():
            self._is_game_over = True
            self.winner = "T"

    def _update_legal_moves(self):
        if self.is_game_over():
            self.legal_moves = []
            return

        legal_moves = []
        for i in range(3):
            for j in range(3):
                if not (self.X[i, j] or self.O[i, j]):
                    legal_moves.append((i, j))
        self.legal_moves = legal_moves
        return

    def print_game(self):
        i = 0
        current_board = np.full((3, 3), "_", dtype=str)
        for move in self.move_stack:
            if i % 2 == 0:
                print(i // 2)
                current_board[move[0], move[1]] = "X"
            else:
                current_board[move[0], move[1]] = "O"
            i += 1
            print(current_board)
        print(f"Winner: {self.winner}")


def move_to_tokens(move):
    return f"{move[0]},{move[1]}"


def tokens_to_move(token, corrected=False):
    """
    if corrected: returns None if token is "START", "END", "X", "O", or "T"
    otherwise raises InvalidMove exception
    """
    try:
        return tuple(map(int, token.split(",")))
    except ValueError:
        if corrected:
            return None
        else:
            raise InvalidMove(
                f"Invalid token for move: {token}. Expected format 'i,j'."
            )


def board_to_sequence(board: TTTBoard, encoder, device, append_end=True):
    moves_list = ["START"]
    for move in board.move_stack:
        moves_list.append(move_to_tokens(move))
    if board.is_game_over() and append_end:
        moves_list.append("END")
        moves_list.append(board.winner)
    encoded = encoder.transform(moves_list)
    return torch.tensor(np.array([encoded]), device=device)


def move_stack_to_sequence(move_stack, encoder, device):
    moves_list = ["START"]
    for move in move_stack:
        moves_list.append(move_to_tokens(move))
    encoded = encoder.transform(moves_list)
    return torch.tensor(np.array([encoded]), device=device)


def sequence_to_move_stack(sequence, encoder):
    tokens = encoder.inverse_transform(sequence)
    move_stack = []
    for token in tokens:
        move = tokens_to_move(token, corrected=True)
        if move is not None:
            move_stack.append(move)
    return move_stack


def tokens_list():
    r = ["START", "END", "X", "O", "T"]
    for i in range(3):
        for j in range(3):
            r.append(f"{i},{j}")
    return r
