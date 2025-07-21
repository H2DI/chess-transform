import torch
import os
import torch.optim as optim


def initialize_optimizer(training_config, model):
    """
    Adam, with warmup and cosine.
    """
    optimizer = optim.Adam(model.parameters(), lr=training_config.lr)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=training_config.warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config.final_lr_time - training_config.warmup,
                eta_min=training_config.lr_min,
            ),
        ],
        milestones=[training_config.warmup],
    )
    return optimizer, scheduler


def train_step(seq, model, criterion, device):
    seq = seq.to(device)

    input_seq = seq[:, :-1]
    target = seq[:, 1:]

    b, T = seq.shape

    tgt_mask = torch.tril(torch.ones(T - 1, T - 1)).to(device).bool()
    logits = model(input_seq, mask=tgt_mask)
    loss = criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))
    return loss, logits


def save_checkpoint(
    model_config,
    training_config,
    n_steps,
    n_games,
    file_number,
    encoder,
    model,
    optimizer,
    scheduler,
):
    checkpoint = {
        "model_config": model_config,
        "training_config": training_config,
        "n_steps": n_steps,
        "n_games": n_games,
        "file_number": file_number,
        "encoder": encoder,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    name = checkpoint["model_config"].name
    os.makedirs(f"checkpoints/{name}", exist_ok=True)
    torch.save(checkpoint, f"checkpoints/{name}/checkpoint_{n_games}.pth")
    print(f"Saved at checkpoint_{n_games}.pth")


def log_grads(writer, model, step):
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm**0.5
    writer.add_scalar("GradNorm/train", grad_norm, step)


def log_weight_norms(writer, model, step):
    with torch.no_grad():
        weight_norm = 0.0
        for param in model.parameters():
            weight_norm += param.data.norm(2).item() ** 2
        weight_norm = weight_norm**0.5
        writer.add_scalar("WeightNorm/train", weight_norm, step)


def log_stat_group(writer, name, values, step):
    writer.add_scalars(
        name,
        {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        },
        step,
    )
