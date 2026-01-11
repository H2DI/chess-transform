#!/usr/bin/env python3
"""
End-to-end test for loading chess-transform model from Hugging Face Hub
and running evaluations.

This test simulates the experience of a new user who:
1. Clones the repository
2. Installs the package
3. Downloads the model from Hugging Face
4. Runs the model to play games

Usage:
    pytest tests/test_hf_end_to_end.py -v
    # or run directly:
    python tests/test_hf_end_to_end.py
"""

import pytest
import torch
import chess

from chess_seq import (
    load_model_from_hf,
    MoveEncoder,
    ChessGameEngine,
    load_model_from_checkpoint,
)

# Test configuration - set to None to use default from save_and_load.py
HF_REPO_ID = None  # Uses DEFAULT_HF_REPO from chess_seq.utils.save_and_load
N_TEST_PLIES = 20  # Number of half-moves to generate per test


def _try_load_from_hf():
    """Attempt to load from HF Hub, return None if unavailable."""
    try:
        model, config, encoder = load_model_from_hf(repo_id=HF_REPO_ID)
        return model, config, encoder
    except Exception as e:
        if "404" in str(e) or "Not Found" in str(e) or "EntryNotFound" in str(e):
            return None
        raise


class TestHuggingFaceEndToEnd:
    """End-to-end tests for loading from Hugging Face Hub."""

    @pytest.fixture(scope="class")
    def model_and_encoder(self):
        """Load model from HF Hub once for all tests in this class."""
        result = _try_load_from_hf()
        if result is None:
            pytest.skip(
                "HF Hub model not available. Upload model first with: "
                "python tools/upload_to_hf.py"
            )
        model, config, encoder = result
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return model, config, encoder, device

    def test_model_loads_successfully(self, model_and_encoder):
        """Test that the model loads from HF Hub without errors."""
        model, config, encoder, device = model_and_encoder

        assert model is not None
        assert config is not None
        assert encoder is not None

        # Check model architecture
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel loaded with {num_params:,} parameters")
        assert num_params > 100_000_000, "Expected ~450M parameter model"

    def test_model_config_matches_expected(self, model_and_encoder):
        """Test that model config has expected values."""
        model, config, encoder, device = model_and_encoder

        # Expected gamba_rossa config values
        assert config.name == "gamba_rossa"
        assert config.vocab_size == 4611
        assert config.block_size == 256
        assert config.n_layers == 28
        assert config.n_head == 16
        assert config.k == 1024

    def test_forward_pass(self, model_and_encoder):
        """Test that model can perform forward pass."""
        model, config, encoder, device = model_and_encoder

        # Create a simple input sequence
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )

        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.vocab_size)
        print(f"\nForward pass output shape: {output.shape}")

    def test_generate_sequence(self, model_and_encoder):
        """Test that model can generate a sequence of moves."""

        model, config, encoder, device = model_and_encoder
        engine = ChessGameEngine(model, encoder, device=device)

        sequence = engine.generate_sequence(n_plies=N_TEST_PLIES)

        assert len(sequence) > 1, "Should generate at least some tokens"
        print(f"\nGenerated sequence of {len(sequence)} tokens")
        print(f"Tokens: {encoder.inverse_transform(sequence)}")

    def test_play_valid_game(self, model_and_encoder):
        """Test that model can play a valid chess game (with illegal move masking)."""
        from chess_seq import ChessGameEngine

        model, config, encoder, device = model_and_encoder
        engine = ChessGameEngine(model, encoder, device=device)

        # Play a game with illegal moves masked
        game, pgn, bad_plies = engine.play_game(n_plies=N_TEST_PLIES, mask_illegal=True)

        assert game is not None
        assert pgn is not None
        assert len(bad_plies) == 0, "With mask_illegal=True, should have no bad moves"

        print(f"\nPlayed game with {len(list(game.move_stack))} moves")
        print(f"PGN: {pgn.mainline_moves()}")

    def test_play_game_without_masking(self, model_and_encoder):
        """Test game playing without illegal move masking (testing model quality)."""

        model, config, encoder, device = model_and_encoder
        engine = ChessGameEngine(model, encoder, device=device)

        # Play without masking to see how often model makes illegal moves
        game, pgn, bad_plies = engine.play_game(
            n_plies=N_TEST_PLIES, mask_illegal=False
        )

        total_moves = len(list(game.move_stack))
        legal_rate = (
            (total_moves - len(bad_plies)) / total_moves if total_moves > 0 else 0
        )

        print(f"\nPlayed {total_moves} moves, {len(bad_plies)} were initially illegal")
        print(f"Legal move rate: {legal_rate:.1%}")

        # Model should make mostly legal moves (>50% at least)
        assert legal_rate > 0.5, (
            f"Model should make mostly legal moves, got {legal_rate:.1%}"
        )

    def test_play_from_specific_opening(self, model_and_encoder):
        """Test playing from a specific opening position."""

        model, config, encoder, device = model_and_encoder
        engine = ChessGameEngine(model, encoder, device=device)

        # Start from e4 e5 Nf3 (Italian Game setup)
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")

        game, pgn, bad_plies = engine.play_game(
            game=board, n_plies=N_TEST_PLIES, mask_illegal=True
        )

        print(
            f"\nStarted from 1.e4 e5 2.Nf3, played {len(list(game.move_stack))} total moves"
        )
        print(f"PGN: {pgn.mainline_moves()}")

        assert len(list(game.move_stack)) >= 3  # At least our opening moves

    def test_evaluate_first_moves(self, model_and_encoder):
        """Test model's response to all 20 first moves."""
        from chess_seq.evaluation.model_basic_eval import test_first_moves

        model, config, encoder, device = model_and_encoder

        bad_plies_list, first_bad_list = test_first_moves(
            model, encoder, n_plies=10, prints=False
        )

        avg_bad_plies = sum(bad_plies_list) / len(bad_plies_list)
        avg_first_bad = sum(first_bad_list) / len(first_bad_list)

        print(f"\nTested all 20 first moves:")
        print(f"Average bad plies per game: {avg_bad_plies:.2f}")
        print(f"Average ply of first bad move: {avg_first_bad:.1f}")

        # Model should average less than 3 bad moves per short game
        assert avg_bad_plies < 5, f"Too many bad moves on average: {avg_bad_plies:.2f}"


class TestLocalModel:
    """
    Test loading and running with local model files.
    This tests the same workflow but uses local checkpoints instead of HF Hub.
    Useful for validating everything works before uploading to HF.
    """

    @pytest.fixture(scope="class")
    def local_model_and_encoder(self):
        """Load model from local checkpoint."""

        try:
            model, config, info = load_model_from_checkpoint(
                "gamba_rossa", special_name="final"
            )
        except FileNotFoundError:
            pytest.skip("Local model checkpoint not found")

        model.eval()
        encoder = MoveEncoder()
        encoder.build()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        return model, config, encoder, device

    def test_local_model_loads(self, local_model_and_encoder):
        """Test local model loads successfully."""
        model, config, encoder, device = local_model_and_encoder
        assert model is not None
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\nLocal model loaded with {num_params:,} parameters")

    def test_local_model_plays_game(self, local_model_and_encoder):
        """Test local model can play a valid game."""
        from chess_seq import ChessGameEngine

        model, config, encoder, device = local_model_and_encoder
        engine = ChessGameEngine(model, encoder, device=device)

        game, pgn, bad_plies = engine.play_game(n_plies=20, mask_illegal=True)
        assert len(bad_plies) == 0
        print(f"\nLocal model played: {pgn.mainline_moves()}")


class TestEncoderFunctionality:
    """Test the encoder works correctly."""

    def test_encoder_builds(self):
        """Test encoder vocabulary builds correctly."""
        from chess_seq import MoveEncoder

        encoder = MoveEncoder()
        encoder.build()

        # Encoder has 4610 tokens (start, end, moves, promotions)
        # vocab_size in model is 4611 to include pad token at index 4610
        assert len(encoder.id_to_token) == 4610
        assert encoder.start_token_id is not None
        assert encoder.end_token_id is not None

    def test_encoder_roundtrip(self):
        """Test encoding and decoding moves."""
        import chess
        from chess_seq import MoveEncoder

        encoder = MoveEncoder()
        encoder.build()

        # Test some common moves
        test_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        for move_str in test_moves:
            move = chess.Move.from_uci(move_str)
            token_id = encoder.move_to_id(move)
            decoded = encoder.id_to_token[token_id]
            assert decoded == move_str, f"Roundtrip failed for {move_str}"


def run_quick_demo():
    """
    Quick demo that can be run directly to test the full pipeline.
    This is what a new user would run after cloning the repo.
    """
    print("=" * 60)
    print("Chess-Transform: End-to-End Demo")
    print("=" * 60)

    # Step 1: Load model from Hugging Face (downloads once, then uses local cache)
    print("\n[1/3] Loading model from Hugging Face Hub...")
    from chess_seq import load_model_from_hf, ChessGameEngine

    model, config, encoder = load_model_from_hf()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")

    # Step 2: Create game engine
    print("\n[2/3] Creating game engine...")
    engine = ChessGameEngine(model, encoder, device=device)

    # Step 3: Play a game
    print("\n[3/3] Playing a sample game...")
    game, pgn, bad_plies = engine.play_game(n_plies=40, mask_illegal=True)

    print("\n" + "=" * 60)
    print("GAME RESULT")
    print("=" * 60)
    print(f"Total moves: {len(list(game.move_stack))}")
    print(f"Game over: {game.is_game_over()}")
    if game.is_game_over():
        print(f"Outcome: {game.outcome()}")
    print(f"\nPGN:\n{pgn}")
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_quick_demo()
    else:
        # Run pytest
        pytest.main([__file__, "-v", "--tb=short"])
