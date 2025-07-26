import torch
import torch.optim as optim

# from chess_seq.training.training_config_classes import TrainingConfig

# globals()["TrainingConfig"] = TrainingConfig


def initialize_optimizer(training_config, model):
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
