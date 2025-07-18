import os
import re
import torch


def get_latest_checkpoint(model_name):
    model_dir = f"checkpoints/{model_name}/"
    pattern = re.compile(r"checkpoint_(\d+)\.pth")
    max_suffix = -1
    max_file = None

    for fname in os.listdir(model_dir):
        match = pattern.match(fname)
        if match:
            suffix = int(match.group(1))
            if suffix > max_suffix:
                max_suffix = suffix
                max_file = fname

    if max_file:
        checkpoint_path = os.path.join(model_dir, max_file)
        print(f"Largest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = None
        print("No checkpoint files found.")

    return checkpoint_path


def save_checkpoint(checkpoint):
    name = checkpoint["model_config"].name
    n_games = checkpoint["n_games"]
    torch.save(checkpoint, f"checkpoints/{name}/checkpoint_{n_games}.pth")


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


# writer.add_scalar(f"{name}/min", min(values), step)
# writer.add_scalar(f"{name}/max", max(values), step)
# writer.add_scalar(f"{name}/avg", sum(values) / len(values), step)
