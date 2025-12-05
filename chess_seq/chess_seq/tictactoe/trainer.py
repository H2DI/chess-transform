from chess_seq.training.trainer_runner import ChessTrainerRunner
from chess_seq.tictactoe.game_engine import TTTGameEngine, TTTGamePlayer
from chess_seq.tictactoe.mechanics import TTTBoard
from chess_seq.training.trainer import log_stat_group, log_average_group


class TTTTrainerRunner(ChessTrainerRunner):
    def __init__(self, session_config, model_config=None, training_config=None):
        super().__init__(session_config, model_config, training_config)

    def _evaluate_model(self):
        self.model.eval()
        eval_legal_moves_and_log(
            self.model,
            self.encoder_dict,
            self.writer,
            self.n_games,
            self.config.test_games_lengths,
        )
        self.model.train()


def eval_legal_moves_and_log(model, encoder, writer, n_games, lengths):
    engine = TTTGameEngine(model, encoder)
    sequence = engine.generate_sequence()
    print(sequence)
    print(encoder.inverse_transform(sequence))

    for n_plies in lengths:
        n_bad, t_first_bad, ends, wins = test_first_moves(
            model, encoder, n_plies=n_plies
        )
        log_stat_group(writer, f"Play{n_plies}/NumberOfBadMoves", n_bad, n_games)
        log_stat_group(writer, f"Play{n_plies}/FirstBadMoves", t_first_bad, n_games)
        log_average_group(writer, f"Play{n_plies}/EndPredictions", ends, n_games)
        log_average_group(writer, f"Play{n_plies}/WinnerPredictions", wins, n_games)


def test_first_moves(model, encoder, n_plies=9, prints=False):
    game = TTTBoard()
    first_moves = list(game.legal_moves)

    all_number_of_bad_plies = []
    all_first_bad_plies = []
    all_end_predictions = []
    all_winner_predictions = []
    for move in first_moves:
        game = TTTBoard()
        game.push(move)

        player = TTTGamePlayer(model, encoder, game, greedy=False)
        player.play_game(n_plies=n_plies)
        bad_plies = player.bad_plies

        if prints:
            print(f"{move=}")
            print(
                f"{len(bad_plies)} bad moves out of {len(game.move_stack)}. "
                + f"First bad ply: {bad_plies[0]}"
                if bad_plies
                else "0 bad moves"
            )

        all_number_of_bad_plies.append(len(bad_plies))
        all_first_bad_plies.append(bad_plies[0] if bad_plies else n_plies)
        all_end_predictions.append(player.end_correctly_predicted)
        all_winner_predictions.append(player.winner_correctly_predicted)
    return (
        all_number_of_bad_plies,
        all_first_bad_plies,
        all_end_predictions,
        all_winner_predictions,
    )
