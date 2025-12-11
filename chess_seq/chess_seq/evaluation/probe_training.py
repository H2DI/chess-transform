"""
Dataset and training utilities for probe training.
"""

import torch
import torch.nn as nn
import chess
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional, Dict, List

from .position_encoder import BoardPositionEncoder
from .activation_extractor import MultiLayerExtractor
from ..encoder import MoveEncoder


class ProbeDataset(Dataset):
    """
    Dataset for probe training.
    Generates activations and board positions from game sequences.
    """

    def __init__(
        self,
        data_file: str,
        model: nn.Module,
        encoder: MoveEncoder,
        layer_idx: int,
        max_samples: Optional[int] = None,
        device: str = "cpu",
        cache_activations: bool = True,
    ):
        """
        Args:
            data_file: Path to .npz file containing encoded games
            model: The ChessNet model to extract activations from
            encoder: MoveEncoder for decoding moves
            layer_idx: Which layer to extract activations from
            max_samples: Maximum number of samples to generate (None = all)
            device: Device to run model on
            cache_activations: Whether to cache activations in memory
        """
        self.encoder = encoder
        self.position_encoder = BoardPositionEncoder()
        self.layer_idx = layer_idx
        self.device = device
        self.cache_activations = cache_activations

        # Load encoded games
        print(f"Loading data from {data_file}...")
        data = np.load(data_file, allow_pickle=True)

        # Handle different data formats
        if "encoded_games" in data:
            # Old format: array of games
            self.encoded_games = data["encoded_games"]
        elif "tokens" in data and "game_ids" in data:
            # New format: flattened tokens with game_ids
            tokens = data["tokens"]
            game_ids = data["game_ids"]

            # Group tokens by game_id
            unique_game_ids = np.unique(game_ids)
            self.encoded_games = []
            for game_id in unique_game_ids:
                game_tokens = tokens[game_ids == game_id]
                if len(game_tokens) == 1 and isinstance(game_tokens[0], np.ndarray):
                    # Token is already an array
                    self.encoded_games.append(game_tokens[0])
                else:
                    self.encoded_games.append(game_tokens)
        else:
            raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")

        print(f"Loaded {len(self.encoded_games)} games")

        # Setup activation extractor
        self.model = model
        self.model.eval()
        self.extractor = MultiLayerExtractor(model, [layer_idx])

        # Generate all samples from games
        print("Generating samples from games...")
        self.samples = self._generate_samples(max_samples)
        print(f"Generated {len(self.samples)} samples")

        # Optionally cache activations
        if cache_activations:
            print("Caching activations...")
            self._cache_all_activations()

    def _generate_samples(self, max_samples: Optional[int]) -> List[Dict]:
        """
        Generate (sequence, board_position) pairs from games.
        Each sample corresponds to a position in a game.
        """
        samples = []

        for game_idx, game in enumerate(
            tqdm(self.encoded_games, desc="Processing games")
        ):
            # Skip start token
            game = game[game != self.encoder.start_token_id]
            game = game[game != self.encoder.end_token_id]

            # Create a board and replay moves
            board = chess.Board()

            # For each position in the game
            for move_idx, token_id in enumerate(game):
                # Store current board position
                board_tensor = self.position_encoder.board_to_tensor(board)

                # Store sequence up to this point (including start token)
                sequence = [self.encoder.start_token_id] + game[: move_idx + 1].tolist()

                samples.append(
                    {
                        "sequence": sequence,
                        "board_state": board_tensor,
                        "game_idx": game_idx,
                        "move_idx": move_idx,
                    }
                )

                # Apply the move
                try:
                    move = self.encoder.id_to_token[int(token_id)]
                    move_obj = self.encoder.token_to_move(move)
                    if move_obj and move_obj in board.legal_moves:
                        board.push(move_obj)
                    else:
                        # Invalid move, skip rest of game
                        break
                except (KeyError, ValueError, chess.IllegalMoveError):
                    # Invalid token or move, skip rest of game
                    break

                if max_samples and len(samples) >= max_samples:
                    return samples

        return samples

    def _cache_all_activations(self):
        """Pre-compute and cache all activations."""
        self.cached_activations = []

        with torch.no_grad():
            for sample in tqdm(self.samples, desc="Caching activations"):
                sequence = torch.tensor([sample["sequence"]], device=self.device)
                acts = self.extractor.extract(sequence)
                # Take the last token's activation
                act = acts[self.layer_idx][:, -1, :]  # (1, D)
                self.cached_activations.append(act.cpu())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        board_state = sample["board_state"]

        if self.cache_activations:
            # Use cached activation
            activation = self.cached_activations[idx]
        else:
            # Compute activation on the fly
            sequence = torch.tensor([sample["sequence"]], device=self.device)
            with torch.no_grad():
                acts = self.extractor.extract(sequence)
                activation = acts[self.layer_idx][:, -1, :].cpu()  # (1, D)

        # Remove batch dimension
        activation = activation.squeeze(0)  # (D,)

        return activation, board_state


def train_probe(
    probe: nn.Module,
    train_dataset: ProbeDataset,
    val_dataset: Optional[ProbeDataset],
    num_epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> Dict:
    """
    Train a probe model.

    Args:
        probe: The probe model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model

    Returns:
        Dictionary with training history
    """
    probe = probe.to(device)
    position_encoder = BoardPositionEncoder()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with hooks
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    # Use cross entropy loss for multi-class classification per square
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_square_acc": [],
        "train_board_acc": [],
        "val_loss": [],
        "val_square_acc": [],
        "val_board_acc": [],
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training
        probe.train()
        train_loss = 0.0
        train_metrics = {"square_accuracy": 0.0, "board_accuracy": 0.0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for activations, board_states in pbar:
            activations = activations.to(device)
            board_states = board_states.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_states = probe(activations)  # (B, 13, 8, 8)

            # Compute loss (cross entropy over the 13 classes per square)
            # Reshape to (B*64, 13) and (B*64,)
            pred_flat = pred_states.permute(0, 2, 3, 1).reshape(-1, 13)  # (B*64, 13)
            target_flat = board_states.argmax(dim=1).reshape(-1)  # (B*64,)

            loss = criterion(pred_flat, target_flat)
            loss.backward()
            optimizer.step()

            # Compute metrics
            with torch.no_grad():
                metrics = position_encoder.compute_accuracy(pred_states, board_states)
                train_metrics["square_accuracy"] += metrics["square_accuracy"]
                train_metrics["board_accuracy"] += metrics["board_accuracy"]
                train_loss += loss.item()
                num_batches += 1

            pbar.set_postfix(
                {
                    "loss": loss.item(),
                    "sq_acc": metrics["square_accuracy"],
                    "bd_acc": metrics["board_accuracy"],
                }
            )

        # Average metrics
        train_loss /= num_batches
        train_metrics["square_accuracy"] /= num_batches
        train_metrics["board_accuracy"] /= num_batches

        history["train_loss"].append(train_loss)
        history["train_square_acc"].append(train_metrics["square_accuracy"])
        history["train_board_acc"].append(train_metrics["board_accuracy"])

        print(
            f"\nEpoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
            f"Square Acc: {train_metrics['square_accuracy']:.4f}, "
            f"Board Acc: {train_metrics['board_accuracy']:.4f}"
        )

        # Validation
        if val_loader:
            probe.eval()
            val_loss = 0.0
            val_metrics = {"square_accuracy": 0.0, "board_accuracy": 0.0}
            num_batches = 0

            with torch.no_grad():
                for activations, board_states in tqdm(val_loader, desc="Validation"):
                    activations = activations.to(device)
                    board_states = board_states.to(device)

                    pred_states = probe(activations)

                    pred_flat = pred_states.permute(0, 2, 3, 1).reshape(-1, 13)
                    target_flat = board_states.argmax(dim=1).reshape(-1)
                    loss = criterion(pred_flat, target_flat)

                    metrics = position_encoder.compute_accuracy(
                        pred_states, board_states
                    )
                    val_metrics["square_accuracy"] += metrics["square_accuracy"]
                    val_metrics["board_accuracy"] += metrics["board_accuracy"]
                    val_loss += loss.item()
                    num_batches += 1

            val_loss /= num_batches
            val_metrics["square_accuracy"] /= num_batches
            val_metrics["board_accuracy"] /= num_batches

            history["val_loss"].append(val_loss)
            history["val_square_acc"].append(val_metrics["square_accuracy"])
            history["val_board_acc"].append(val_metrics["board_accuracy"])

            print(
                f"Val Loss: {val_loss:.4f}, "
                f"Square Acc: {val_metrics['square_accuracy']:.4f}, "
                f"Board Acc: {val_metrics['board_accuracy']:.4f}"
            )

            # Save best model
            if save_path and val_metrics["square_accuracy"] > best_val_acc:
                best_val_acc = val_metrics["square_accuracy"]
                torch.save(
                    {
                        "epoch": epoch,
                        "probe_state_dict": probe.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "square_accuracy": best_val_acc,
                        "history": history,
                    },
                    save_path,
                )
                print(f"Saved best model with square accuracy: {best_val_acc:.4f}")

    return history


def evaluate_probe(
    probe: nn.Module,
    dataset: ProbeDataset,
    device: str = "cpu",
    num_samples_to_visualize: int = 5,
):
    """
    Evaluate a probe and visualize some predictions.

    Args:
        probe: Trained probe model
        dataset: Dataset to evaluate on
        device: Device to run on
        num_samples_to_visualize: Number of samples to print
    """
    probe = probe.to(device)
    probe.eval()
    position_encoder = BoardPositionEncoder()

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_metrics = {"square_accuracy": 0.0, "board_accuracy": 0.0}
    num_batches = 0

    with torch.no_grad():
        for activations, board_states in tqdm(loader, desc="Evaluating"):
            activations = activations.to(device)
            board_states = board_states.to(device)

            pred_states = probe(activations)
            metrics = position_encoder.compute_accuracy(pred_states, board_states)

            all_metrics["square_accuracy"] += metrics["square_accuracy"]
            all_metrics["board_accuracy"] += metrics["board_accuracy"]
            num_batches += 1

    all_metrics["square_accuracy"] /= num_batches
    all_metrics["board_accuracy"] /= num_batches

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Square Accuracy: {all_metrics['square_accuracy']:.4f}")
    print(f"Board Accuracy: {all_metrics['board_accuracy']:.4f}")
    print("=" * 60)

    # Visualize some examples
    print("\nVisualizing sample predictions:")
    for i in range(min(num_samples_to_visualize, len(dataset))):
        activation, board_state = dataset[i]
        activation = activation.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_state = probe(activation).cpu().squeeze(0)

        print(f"\n{'=' * 60}")
        print(f"Sample {i + 1}")
        print(f"{'=' * 60}")
        position_encoder.visualize_prediction(pred_state, board_state)

    return all_metrics
