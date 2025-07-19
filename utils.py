import os
import re
import torch

import chess_seq.models as models


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


def load_model(model_name):
    """
    Loads model for inference.
    Return model, encoder, checkpoint
    """
    checkpoint_path = get_latest_checkpoint(model_name)
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )

    model_config = checkpoint["model_config"]
    model = models.ChessNet(model_config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoder = checkpoint["encoder"]
    return model, encoder, checkpoint


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
