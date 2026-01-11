"""
Board position encoder for probe training.
Converts chess board states into tensor representations.
"""

import chess
import torch


class BoardPositionEncoder:
    """
    Encodes chess board positions into tensor format for probe training.

    Representation: 8x8x13 tensor where:
    - First 12 channels represent the 12 piece types (6 white + 6 black)
    - 13th channel represents empty squares

    Piece order: P, N, B, R, Q, K, p, n, b, r, q, k (white then black)
    """

    def __init__(self):
        # Map piece symbols to channel indices
        self.piece_to_channel = {
            "P": 0,
            "N": 1,
            "B": 2,
            "R": 3,
            "Q": 4,
            "K": 5,
            "p": 6,
            "n": 7,
            "b": 8,
            "r": 9,
            "q": 10,
            "k": 11,
        }
        self.num_channels = 13  # 12 pieces + 1 empty

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert a chess.Board to a tensor representation.

        Args:
            board: chess.Board object

        Returns:
            torch.Tensor of shape (13, 8, 8)
        """
        tensor = torch.zeros(self.num_channels, 8, 8, dtype=torch.float32)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            row = square // 8
            col = square % 8

            if piece is None:
                tensor[12, row, col] = 1.0  # Empty square
            else:
                channel = self.piece_to_channel[piece.symbol()]
                tensor[channel, row, col] = 1.0

        return tensor

    def tensor_to_board(self, tensor: torch.Tensor) -> chess.Board:
        """
        Convert a tensor representation back to a chess.Board.
        Useful for visualization and verification.

        Args:
            tensor: torch.Tensor of shape (13, 8, 8)

        Returns:
            chess.Board object
        """
        board = chess.Board.empty()

        channel_to_piece = {v: k for k, v in self.piece_to_channel.items()}

        for row in range(8):
            for col in range(8):
                square = row * 8 + col

                # Find which channel has the highest value across all channels
                channel_values = tensor[:, row, col]
                channel_idx = int(channel_values.argmax().item())

                # If argmax is the empty channel (index 12), leave square empty
                if channel_idx == 12:
                    continue

                # Otherwise set the predicted piece
                piece_symbol = channel_to_piece.get(channel_idx)
                if piece_symbol is not None:
                    piece = chess.Piece.from_symbol(piece_symbol)
                    board.set_piece_at(square, piece)

        return board

    def compute_accuracy(
        self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> dict:
        """
        Compute various accuracy metrics between predicted and target board states.

        Args:
            pred_tensor: Predicted tensor of shape (13, 8, 8) or (B, 13, 8, 8)
            target_tensor: Target tensor of shape (13, 8, 8) or (B, 13, 8, 8)

        Returns:
            Dictionary with accuracy metrics
        """
        # Handle batched inputs
        if pred_tensor.dim() == 4:
            # Take argmax over channels to get predicted piece type per square
            pred_pieces = pred_tensor.argmax(dim=1)  # (B, 8, 8)
            target_pieces = target_tensor.argmax(dim=1)  # (B, 8, 8)

            # Square accuracy: fraction of squares with correct piece
            square_accuracy = (pred_pieces == target_pieces).float().mean().item()

            # Board accuracy: fraction of boards that are perfectly correct
            board_correct = (pred_pieces == target_pieces).all(dim=(1, 2))
            board_accuracy = board_correct.float().mean().item()

            # Per-piece accuracy
            piece_accuracies = {}
            for piece_name, channel_idx in self.piece_to_channel.items():
                mask = target_pieces == channel_idx
                if mask.sum() > 0:
                    correct = ((pred_pieces == channel_idx) & mask).sum().float()
                    total = mask.sum().float()
                    piece_accuracies[piece_name] = (correct / total).item()

        else:
            # Single board
            pred_pieces = pred_tensor.argmax(dim=0)  # (8, 8)
            target_pieces = target_tensor.argmax(dim=0)  # (8, 8)

            square_accuracy = (pred_pieces == target_pieces).float().mean().item()
            board_accuracy = float(torch.all(pred_pieces == target_pieces).item())

            piece_accuracies = {}
            for piece_name, channel_idx in self.piece_to_channel.items():
                mask = target_pieces == channel_idx
                if mask.sum() > 0:
                    correct = ((pred_pieces == channel_idx) & mask).sum().float()
                    total = mask.sum().float()
                    piece_accuracies[piece_name] = (correct / total).item()

        return {
            "square_accuracy": square_accuracy,
            "board_accuracy": board_accuracy,
            "piece_accuracies": piece_accuracies,
        }

    def visualize_prediction(
        self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor
    ):
        """
        Print a side-by-side comparison of predicted and target boards.

        Args:
            pred_tensor: Predicted tensor of shape (13, 8, 8)
            target_tensor: Target tensor of shape (13, 8, 8)
        """
        pred_board = self.tensor_to_board(pred_tensor)
        target_board = self.tensor_to_board(target_tensor)

        print("Predicted Board:")
        print(pred_board)
        print("\nTarget Board:")
        print(target_board)
        print("\nAccuracy:", self.compute_accuracy(pred_tensor, target_tensor))


if __name__ == "__main__":
    # Test the encoder
    encoder = BoardPositionEncoder()

    # Test with starting position
    board = chess.Board()
    tensor = encoder.board_to_tensor(board)
    print("Tensor shape:", tensor.shape)
    print("Sum per channel (should be 16 for pieces, 32 for empty):")
    print(tensor.sum(dim=(1, 2)))

    # Test reconstruction
    reconstructed = encoder.tensor_to_board(tensor)
    print("\nOriginal board:")
    print(board)
    print("\nReconstructed board:")
    print(reconstructed)
    print("\nBoards match:", board.fen() == reconstructed.fen())

    # Test accuracy computation
    accuracy = encoder.compute_accuracy(tensor, tensor)
    print("\nSelf-accuracy (should be 1.0):", accuracy["square_accuracy"])
