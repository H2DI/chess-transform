import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import asdict
from .config import Config
import numpy as np


from .dataset import ChessTRMDataset
from .core import TinyRecursiveChessModel, DeepSupervision
from torch.utils.data import random_split, DataLoader
from chess_seq.encoder import MoveEncoder
from torch.utils.tensorboard import SummaryWriter
import os
import logging


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def train_one_epoch(
    superviser: DeepSupervision,
    cfg: Config,
    loader: DataLoader,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    global_step_start: int = 0,
    val_loader: Optional[DataLoader] = None,
):
    total_loss = 0.0
    total_examples = 0
    batches_seen = 0
    global_step = global_step_start

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        # DeepSupervision wrapper handles forward/backward/step
        losses = superviser(x, y)
        loss = np.mean(losses)

        total_examples += x.shape[0]

        # increment steps and compute global step
        batches_seen += 1
        global_step += superviser.N

        # batch-level logging
        if batches_seen % cfg.train.log_interval == 0:
            writer.add_scalar("loss/train_batch", loss, global_step)
            lr = (
                superviser.opt.param_groups[0]["lr"]
                if len(superviser.opt.param_groups) > 0
                else 0.0
            )
            writer.add_scalar("lr", lr, global_step)
            logger.info(
                f"[Epoch {epoch}] Step {global_step} train_loss={loss:.4f} lr={lr:.6g}"
            )

        # Optionally run a quick validation eval every `val_interval` batches
        if batches_seen % cfg.train.val_interval == 0:
            val_loss_sum = 0.0
            val_correct = 0
            val_examples = 0
            with torch.no_grad():
                it = iter(val_loader)
                for vb in range(cfg.train.val_batches):
                    vx, vy = next(it)
                    vx = vx.to(device)
                    vy = vy.to(device)
                    _, _, v_logits, _ = superviser.base_model(vx)
                    v_loss = F.cross_entropy(v_logits, vy, reduction="sum").item()
                    preds = v_logits.argmax(dim=-1)
                    val_loss_sum += v_loss
                    val_correct += (preds == vy).sum().item()
                    val_examples += vx.size(0)

            val_avg = val_loss_sum / val_examples
            val_acc = val_correct / val_examples
            writer.add_scalar("loss/val_batch", val_avg, global_step)
            writer.add_scalar("accuracy/val_batch", val_acc, global_step)
            logger.info(
                f"[Epoch {epoch}] Step {global_step} val_loss={val_avg:.4f} val_acc={val_acc:.4f}"
            )

        # checkpointing
        if global_step % cfg.train.checkpoint_steps == 0:
            os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(cfg.train.checkpoint_dir, f"ckpt_{global_step}.pt")
            torch.save(
                {
                    "model_state": superviser.base_model.state_dict(),
                    "optim_state": superviser.opt.state_dict(),
                    "config": asdict(cfg),
                    "global_step": global_step,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            writer.add_text("checkpoint", f"Saved {ckpt_path}", global_step)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    avg = total_loss / max(1, total_examples)
    return avg, global_step


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

        _, _, logits, _ = model(x)
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
    cfg = Config()

    full_ds = ChessTRMDataset(cfg.data.file)
    N_data = len(full_ds)
    val_n = int(N_data * cfg.data.val_split)
    train_n = N_data - val_n

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
    superviser = DeepSupervision(
        base_model=model,
        opt=optim,
        N=cfg.train.supervision_N,
        aux_loss_weight=1.0,
    )

    # TensorBoard logging setup
    tb_logdir = cfg.train.log_dir
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    steps = 0
    for epoch in range(cfg.train.epochs):
        # Train for one epoch (uses train_one_epoch above) and get steps done
        train_loss, steps = train_one_epoch(
            superviser,
            cfg,
            train_loader,
            device,
            writer=writer,
            global_step_start=steps,
            epoch=epoch,
            val_loader=val_loader,
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
        for name, param in model.named_parameters():
            try:
                writer.add_histogram(name, param.detach().cpu().numpy(), epoch)
            except Exception:
                # Skip parameters that can't be converted to numpy
                pass

    writer.close()


if __name__ == "__main__":
    main()
