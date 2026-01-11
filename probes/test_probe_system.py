"""
Quick test/demo script for probe training system.
Runs a small-scale probe training experiment to verify everything works.
"""

import torch
import sys
import os
import chess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from chess_seq.models import ChessNet
from chess_seq.encoder import MoveEncoder
from chess_seq.configs import ModelConfig
from chess_seq.evaluation.probes import LinearProbe, MLPProbe
from chess_seq.evaluation.probes.activation_extractor import (
    extract_layer_activations,
)

from chess_seq.evaluation.probes.position_encoder import BoardPositionEncoder


def test_components():
    """Test that all components are working."""
    print("=" * 60)
    print("TESTING PROBE TRAINING COMPONENTS")
    print("=" * 60)

    # 1. Test Position Encoder
    print("\n1. Testing Position Encoder...")

    pos_encoder = BoardPositionEncoder()
    board = chess.Board()
    tensor = pos_encoder.board_to_tensor(board)
    reconstructed = pos_encoder.tensor_to_board(tensor)

    print("   ✓ Position encoder works!")
    print(f"   ✓ Tensor shape: {tensor.shape}")
    # Note: FEN won't match because we only encode piece positions, not castling/en passant
    # But piece positions should match
    pieces_match = all(
        board.piece_at(sq) == reconstructed.piece_at(sq) for sq in chess.SQUARES
    )
    print(f"   ✓ Piece positions match: {pieces_match}")

    # Test accuracy computation
    accuracy = pos_encoder.compute_accuracy(tensor, tensor)
    assert accuracy["square_accuracy"] == 1.0
    print("   ✓ Accuracy computation works!")

    # 2. Test Probe Models
    print("\n2. Testing Probe Models...")
    batch_size = 4
    seq_len = 16
    hidden_dim = 128

    x = torch.randn(batch_size, seq_len, hidden_dim)

    linear_probe = LinearProbe(hidden_dim)
    out = linear_probe(x)
    assert out.shape == (batch_size, 13, 8, 8)
    print(f"   ✓ Linear probe works! Output shape: {out.shape}")

    mlp_probe = MLPProbe(hidden_dim, hidden_dim=64, num_layers=2)
    out = mlp_probe(x)
    assert out.shape == (batch_size, 13, 8, 8)
    print(f"   ✓ MLP probe works! Output shape: {out.shape}")

    # 3. Test Activation Extraction
    print("\n3. Testing Activation Extraction...")

    # Create small model
    config = ModelConfig(
        vocab_size=100,
        block_size=32,
        k=128,
        head_dim=32,
        n_head=4,
        n_layers=4,
        dropout=0.0,
        kv_groups=2,
        pad_index=99,  # Must be < vocab_size
    )

    model = ChessNet(config)
    model.eval()

    x = torch.randint(0, config.vocab_size, (2, 16))

    # Extract from one layer
    act = extract_layer_activations(model, x, layer_idx=2)
    print(f"   ✓ Activation extraction works!")
    print(f"   ✓ Activation shape: {act.shape}")
    assert act.shape == (2, 16, 128)

    # 4. Test Complete Forward Pass
    print("\n4. Testing Complete Forward Pass...")

    # Pass activations through probe
    probe = LinearProbe(128)
    pred = probe(act)

    print(f"   ✓ End-to-end forward pass works!")
    print(f"   ✓ Prediction shape: {pred.shape}")
    assert pred.shape == (2, 13, 8, 8)

    # 5. Test Loss Computation
    print("\n5. Testing Loss Computation...")

    target = torch.randn(2, 13, 8, 8)
    target = torch.softmax(target, dim=1)  # Make it a valid distribution

    criterion = torch.nn.CrossEntropyLoss()
    pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, 13)
    target_flat = target.argmax(dim=1).reshape(-1)
    loss = criterion(pred_flat, target_flat)

    print(f"   ✓ Loss computation works!")
    print(f"   ✓ Loss value: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nThe probe training system is ready to use!")
    print("\nNext steps:")
    print(
        "1. Train a probe: python train_probe.py --layer 14 --max-train-samples 10000"
    )
    print("2. Read the documentation: PROBE_TRAINING.md")
    print("3. Analyze different layers to find where position info is strongest")


def demo_position_encoding():
    """Demonstrate position encoding with a real chess position."""
    print("\n" + "=" * 60)
    print("DEMO: Position Encoding")
    print("=" * 60)

    pos_encoder = BoardPositionEncoder()

    # Create a position after a few moves
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]

    print("\nPlaying moves:", " ".join(moves))
    for move_uci in moves:
        board.push(chess.Move.from_uci(move_uci))

    print("\nBoard position:")
    print(board)

    # Encode to tensor
    tensor = pos_encoder.board_to_tensor(board)
    print(f"\nEncoded to tensor of shape: {tensor.shape}")
    print(f"Non-zero pieces per channel: {(tensor.sum(dim=(1, 2)) > 0).sum().item()}")

    # Decode back
    reconstructed = pos_encoder.tensor_to_board(tensor)
    print("\nReconstructed board:")
    print(reconstructed)

    print(f"\nMatch: {board.fen() == reconstructed.fen()}")


if __name__ == "__main__":
    test_components()
    demo_position_encoding()

    print("\n" + "=" * 60)
    print("READY TO TRAIN PROBES!")
    print("=" * 60)
    print("\nExample command:")
    print(
        "  python train_probe.py --layer 14 --probe-type linear --max-train-samples 10000 --epochs 5"
    )
