import torch
import torch.nn as nn

from configs import TrainingConfig


def initialize_optimizer(training_config: TrainingConfig, model: nn.Module):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=training_config.wd,
    )

    total_steps = training_config.total_games // training_config.batch_size
    n_warmup = int(training_config.warmup * total_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=n_warmup,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps - n_warmup,
                eta_min=training_config.lr_min,
            ),
        ],
        milestones=[n_warmup],
    )
    return optimizer, scheduler


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


def log_average_group(writer, name, values, step):
    writer.add_scalar(
        name + "avg",
        sum(values) / len(values),
        step,
    )
