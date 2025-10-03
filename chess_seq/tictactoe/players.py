import random
import numpy as np
import torch

from chess_seq.tictactoe.game_engine import TTTGameEngine
import chess_seq.tictactoe.mechanics as mechanics

"""
Players are used in RL environments. 

"""


class Player:
    def get_move(self, game, greedy=False):
        pass


class RandomPlayer(Player):
    def get_move(self, game: mechanics.TTTBoard, greedy=False):
        legal_moves = game.legal_moves
        if not legal_moves:
            return None, False
        return random.choice(legal_moves)


class SimplePlayer(Player):
    def get_move(self, game: mechanics.TTTBoard, greedy=False):
        legal_moves = game.legal_moves
        if not legal_moves:
            return None, False
        return legal_moves[0]


class NNPlayer(Player):
    def __init__(self, model, encoder, mask_illegal=True, device=None):
        self.engine = TTTGameEngine(model, encoder, device=device)
        self.engine.model.eval()
        self.mask_illegal = mask_illegal

    def build_mask(self, game):
        mask = torch.zeros(self.engine.vocab_size, device=self.engine.device)
        if not (self.mask_illegal):
            return mask
        else:
            mask = mask + float("-inf")
            for move in game.legal_moves:
                token = mechanics.move_to_tokens(move)
                tokenid = self.engine.encoder.transform([token])[0]
                mask[tokenid] = 0.0
            return mask

    def get_move(self, game: mechanics.TTTBoard, greedy=True):
        """ """
        sequence = mechanics.board_to_sequence(
            game, self.engine.encoder, self.engine.device
        )
        mask = self.build_mask(game)

        tokenid, _ = self.engine.generate_next_tokenid(
            sequence, greedy=greedy, mask=mask
        )
        token = self.engine.encoder.inverse_transform(tokenid[0])[0]
        move = mechanics.tokens_to_move(token)
        return move

    def predict_end(self, game):
        sequence = mechanics.board_to_sequence(
            game, self.engine.encoder, self.engine.device
        )
        tokenid, _ = self.engine.generate_next_tokenid(sequence, greedy=True)
        token = self.engine.encoder.inverse_transform(tokenid[0])[0]
        return token

    def predict_winner(self, game):
        sequence = mechanics.board_to_sequence(
            game, self.engine.encoder, self.engine.device
        )
        sequence = torch.cat(
            (
                sequence,
                torch.tensor([[self.engine.end_tokenid]], device=self.engine.device),
            ),
            dim=1,
        )
        tokenid, _ = self.engine.generate_next_tokenid(
            sequence, greedy=True, predict_winner=True
        )
        token = self.engine.encoder.inverse_transform(tokenid[0])[0]
        return token


class PlayVPlay:
    """
    Assumes both players play only legal moves.
    """

    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
        self.games = []

    def _play_one_move(self):
        player = self.player1 if self.game.turn % 2 == 0 else self.player2
        move = player.get_move(self.game)
        assert move is not None, "Player returned an illegal move."

        self.game.push(move)
        print(f"Game state after move:\n{self.game.X - self.game.O}\n")
        if self.game.is_game_over():
            print(f"Game over! Winner: {self.game.winner}")

    def play_game(self):
        self.current_game = mechanics.TTTBoard()
        print("New game started.")
        while not self.game.is_game_over():
            self._play_one_move()
        self.games.append(self.game)


class PlayAgainstHuman:
    def __init__(self, player: Player, start=True):
        self.player = player
        self.game = mechanics.TTTBoard()
        if not start:
            print("You plays as O")
            self.adversary_move()
        else:
            print("You plays as X")

    def play(self, move):
        if move not in self.game.legal_moves:
            print(f"Illegal move: {move}. Legal moves are: {self.game.legal_moves}")
        else:
            self.game.push(move)
            print(f"Game state after your move:\n{self.game.X - self.game.O}\n")
            if self.game.is_game_over():
                print(f"Game over! Winner: {self.game.winner}")
            else:
                self._adversary_plays()

    def _adversary_plays(self):
        move = self.player.get_move(self.game)
        print(f"Adversary move: {move}")
        self.game.push(move)
        print(f"Game state after adversary move:\n{self.game.X - self.game.O}\n")
        if self.game.is_game_over():
            print(f"Game over! Winner: {self.game.winner}")


class GamesRecorder:
    """
    Records games by tokenid
    """

    def __init__(self, encoder, player):
        self.encoder = encoder
        self.games = []
        self.player = player

    def play_game(self, print_result=False):
        game = mechanics.TTTBoard()
        seq = ["START"]
        while not game.is_game_over():
            move, _ = self.player.get_play(game)
            game.push(move)
            seq.append(mechanics.move_to_tokens(move))
        seq.append("END")
        seq.append(game.winner)
        self.games.append(self.encoder.transform(seq))
        if print_result:
            print(f"Game finished: {game.move_stack}, Winner: {game.winner}")
            print(game.X - game.O)
            print(game.X + game.O)

    def save_games(self, path):
        with open(path, "wb") as f:
            np.savez(f, *self.games)
