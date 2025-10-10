import numpy as np
import torch
import random

import chess_seq.tictactoe.mechanics as mechanics


class TTTGameEngine:
    def __init__(self, model, encoder, device=None):
        self.model = model
        self.encoder = encoder
        self.device = device if device else next(model.parameters()).device
        self.model.to(self.device)

        self.end_tokenid = encoder.transform(["END"])[0]
        self.special_tokenids = encoder.transform(["START", "END", "X", "O", "T"])
        self.winner_tokenids = encoder.transform(["X", "O", "T"])
        self.vocab_size = len(encoder.classes_)

    @torch.no_grad()
    def generate_sequence(self, sequence=None, n_plies=11):
        """
        outputs a sequence of tokens, regardless of whether this gives a proper
        chess game
        """
        model = self.model
        encoder = self.encoder
        device = self.device
        final_evaluation = False

        if sequence is None:
            sequence = torch.tensor(
                np.array([encoder.transform(["START"])]), device=device
            )
        for _ in range(n_plies):
            out = model(sequence)  # B, T, vocab_size
            next_move = out[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            sequence = torch.cat((sequence, next_move), dim=1)
            if final_evaluation:
                break
            if sequence[0, -1].cpu().numpy() == self.end_tokenid:
                final_evaluation = True

        return sequence.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def generate_next_tokenid(
        self, sequence, greedy=True, predict_winner=False, mask=None
    ):
        candidate_sequence = sequence.clone()
        out = self.model(candidate_sequence)  # B, T, vocab_size
        logits = out[0, -1]  # vocab_size
        if mask is not None:
            logits = logits + mask
        if greedy:
            next_tokenid = logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            entropy = 0
        else:
            if predict_winner:
                # set logits to zero for all tokens that are not winners
                pass
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().item()
            next_tokenid = dist.sample().unsqueeze(0).unsqueeze(0)
        candidate_sequence = torch.cat((candidate_sequence, next_tokenid), dim=1)
        return next_tokenid.cpu().numpy(), candidate_sequence, entropy


class TTTGamePlayer:
    def __init__(self, model, encoder, game=None, device=None, greedy=True):
        self.engine = TTTGameEngine(model, encoder, device)
        self.greedy = greedy

        if game is None:
            self.game = mechanics.TTTBoard()
            self.sequence = torch.tensor(
                np.array([encoder.transform(["START"])]), device=device
            )
        else:
            self.game = game
            self.sequence = mechanics.board_to_sequence(game, encoder, device)

        self.bad_plies = []

    @torch.no_grad()
    def play_game(self, n_plies=9):
        """
        Current behavior: predicts any token, then plays the move corresponding if it is legal, otherwise plays a random legal move.

        When game is over, predicts two more tokens: "END" and the winner.
        """
        self.current_ply = len(self.game.move_stack)

        for _ in range(n_plies):
            tokenid, candidate_sequence, _ = self.engine.generate_next_tokenid(
                self.sequence, greedy=self.greedy, predict_winner=False
            )
            self.current_ply += 1

            self._play_move_or_fallback(
                tokenid,
                candidate_sequence,
            )

            if self.game.is_game_over():
                break

        self._predict_end()
        self._predict_winner()

    def _predict_end(self):
        tokenid, candidate_sequence, _ = self.engine.generate_next_tokenid(
            self.sequence, greedy=self.greedy, predict_winner=False
        )
        self.current_ply += 1
        if tokenid == self.engine.end_tokenid:
            self.end_correctly_predicted = True
            self.sequence = candidate_sequence
        else:
            self.bad_plies.append(self.current_ply)
            self.end_correctly_predicted = False
            self.sequence = torch.cat(
                (
                    self.sequence,
                    torch.tensor(
                        [[self.engine.end_tokenid]], device=self.engine.device
                    ),
                ),
                dim=1,
            )

    def _predict_winner(self):
        self.current_ply += 1
        winner = self.game.winner
        tokenid, _, _ = self.engine.generate_next_tokenid(
            self.sequence, greedy=self.greedy, predict_winner=True
        )
        if winner == self.engine.encoder.inverse_transform([tokenid])[0]:
            self.winner_correctly_predicted = True
        else:
            self.winner_correctly_predicted = False
            self.bad_plies.append(self.current_ply)

    def _play_move_or_fallback(
        self,
        tokenid,
        candidate_sequence,
    ):
        """ """
        legal, true_move_played = self._play_from_tokens(tokenid)

        if not legal:
            self.bad_plies.append(self.current_ply)
            tokens = np.array(
                [
                    self.engine.encoder.transform(
                        [mechanics.move_to_tokens(true_move_played)]
                    )
                ]
            )
            self.sequence = torch.cat(
                (self.sequence, torch.tensor(tokens, device=self.engine.device)),
                dim=1,
            )
        else:
            self.sequence = candidate_sequence

    @torch.no_grad()
    def _play_from_tokens(self, tokenid):
        """ """
        assert not self.game.is_game_over(), "Game is already over."
        try:
            if tokenid in self.engine.special_tokenids:
                raise mechanics.InvalidMove("Special token encountered during play.")
            move = mechanics.tokens_to_move(
                self.engine.encoder.inverse_transform([tokenid])[0]
            )
            if move not in self.game.legal_moves:
                raise mechanics.InvalidMove(f"Illegal move: {move}.")
            else:
                self.game.push(move)
                return True, move
        except mechanics.InvalidMove as e:
            legal_moves = list(self.game.legal_moves)
            random_move = random.choice(legal_moves)
            self.game.push(random_move)
            return False, random_move
