import torch
import torch.nn.functional as F
from typing import Tuple
from config import cfg


from .dataset import ChessTRMDataset
from .core import TinyRecursiveChessModel
from torch.utils.data import random_split, DataLoader
from chess_seq.encoder import MoveEncoder


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    deep_supervision: bool = True,
    supervision_decay: float = 1.0,
):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        if deep_supervision:
            logits_all = model(x, return_all_steps=True)  # (H, B, V)
            H, B, V = logits_all.shape
            loss = 0.0
            weight = 1.0
            weight_sum = 0.0

            for h in range(H):
                logits_h = logits_all[h]  # (B, V)
                loss_h = F.cross_entropy(logits_h, y)
                loss = loss + weight * loss_h
                weight_sum += weight
                weight *= supervision_decay  # can be < 1.0
            loss = loss / weight_sum
        else:
            logits = model(x, return_all_steps=False)  # (B, V)
            loss = F.cross_entropy(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_examples += bs

    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, return_all_steps=False)  # (B, V)
        loss = F.cross_entropy(logits, y)

        preds = logits.argmax(dim=-1)
        correct = (preds == y).sum().item()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += correct
        total_examples += bs

    avg_loss = total_loss / max(1, total_examples)
    acc = total_correct / max(1, total_examples)
    return avg_loss, acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_ds = ChessTRMDataset(cfg.data.file)
    N = len(full_ds)
    val_n = int(N * cfg.data.val_split)
    train_n = N - val_n

    train_ds, val_ds = random_split(full_ds, [train_n, val_n])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=4,
    )

    move_encoder = MoveEncoder()
    move_encoder.load("data/move_encoder.pkl")

    model = TinyRecursiveChessModel(
        token_vocab_size=int(full_ds.X.max()) + 1,
        num_moves=move_encoder.id_to_token.__len__(),
        dim=cfg.model.dim,
        H_cycles=cfg.model.H_cycles,
        L_cycles=cfg.model.L_cycles,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    for epoch in range(cfg.optim.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_loss += F.cross_entropy(logits, y, reduction="sum").item()
                val_correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)

        print(
            f"Epoch {epoch + 1}: val_loss={val_loss / total:.4f}, val_acc={val_correct / total:.4f}"
        )


if __name__ == "__main__":
    main()
