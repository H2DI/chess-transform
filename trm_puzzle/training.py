import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from .config import cfg


from .dataset import ChessTRMDataset
from .core import TinyRecursiveChessModel
from torch.utils.data import random_split, DataLoader
from chess_seq.encoder import MoveEncoder
from torch.utils.tensorboard import SummaryWriter
import os


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    deep_supervision: bool = True,
    supervision_decay: float = 1.0,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    log_interval: int = 100,
    global_step_start: int = 0,
    max_steps: Optional[int] = None,
    val_loader: Optional[DataLoader] = None,
    val_interval: Optional[int] = None,
    val_batches: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_steps: Optional[int] = None,
):
    model.train()
    total_loss = 0.0
    total_examples = 0
    steps_done = 0

    for i, (x, y) in enumerate(loader):
        if max_steps is not None and steps_done >= max_steps:
            break
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
        # increment steps and compute global step
        steps_done += 1
        global_step = global_step_start + steps_done

        # batch-level logging
        if writer is not None and (steps_done) % log_interval == 0:
            writer.add_scalar("loss/train_batch", loss.item(), global_step)
            lr = (
                optimizer.param_groups[0]["lr"]
                if len(optimizer.param_groups) > 0
                else 0.0
            )
            writer.add_scalar("lr", lr, global_step)

        # Optionally run a quick validation eval every `val_interval` batches
        if steps_done % val_interval == 0:
            # evaluate on a small number of validation batches
            val_loss_sum = 0.0
            val_correct = 0
            val_examples = 0
            with torch.no_grad():
                it = iter(val_loader)
                for vb in range(val_batches):
                    vx, vy = next(it)
                    vx = vx.to(device)
                    vy = vy.to(device)
                    v_logits = model(vx, return_all_steps=False)
                    v_loss = F.cross_entropy(v_logits, vy, reduction="sum").item()
                    preds = v_logits.argmax(dim=-1)
                    val_loss_sum += v_loss
                    val_correct += (preds == vy).sum().item()
                    val_examples += vx.size(0)

            if val_examples > 0:
                val_avg = val_loss_sum / val_examples
                val_acc = val_correct / val_examples
                writer.add_scalar("loss/val_batch", val_avg, global_step)
                writer.add_scalar("accuracy/val_batch", val_acc, global_step)

        # checkpointing
        if global_step % checkpoint_steps == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{global_step}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "config": cfg.model,
                    "global_step": global_step,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            if writer is not None:
                writer.add_text("checkpoint", f"Saved {ckpt_path}", global_step)

        steps_done += 1

    avg = total_loss / max(1, total_examples)
    return avg, steps_done


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
    num_moves = move_encoder.id_to_token.__len__()

    model = TinyRecursiveChessModel(
        token_vocab_size=int(full_ds.X.max()) + 1,
        num_moves=num_moves,
        dim=cfg.model.dim,
        H_cycles=cfg.model.H_cycles,
        L_cycles=cfg.model.L_cycles,
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    # TensorBoard logging setup
    tb_logdir = cfg.train.checkpoint_dir + "/tb_logs"
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    # read runtime config from env
    log_interval = cfg.train.log_interval
    val_interval = cfg.train.val_interval
    val_batches = cfg.train.val_batches
    checkpoint_steps = cfg.train.checkpoint_steps
    checkpoint_dir = cfg.train.checkpoint_dir

    for epoch in range(cfg.train.epochs):
        # Train for one epoch (uses train_one_epoch above) and get steps done
        train_loss, steps = train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            deep_supervision=True,
            supervision_decay=1.0,
            writer=writer,
            epoch=epoch,
            log_interval=log_interval,
            val_loader=val_loader,
            val_interval=val_interval,
            val_batches=val_batches,
            checkpoint_dir=checkpoint_dir,
            checkpoint_steps=checkpoint_steps,
        )

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Learning rate (first param group)
        lr = optim.param_groups[0]["lr"] if len(optim.param_groups) > 0 else 0.0

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # Log epoch-level scalars to TensorBoard
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("accuracy/val", val_acc, epoch)
        writer.add_scalar("lr", lr, epoch)

        # Occasionally log parameter histograms
        if (epoch + 1) % 5 == 0:
            for name, param in model.named_parameters():
                try:
                    writer.add_histogram(name, param.detach().cpu().numpy(), epoch)
                except Exception:
                    # Skip parameters that can't be converted to numpy
                    pass

    writer.close()


if __name__ == "__main__":
    main()
