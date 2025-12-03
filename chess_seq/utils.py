import os
import re
import torch

import chess_seq.models as models


def build_and_save_model(model_config):
    model = models.ChessNet(config=model_config)
    base_checkpoint = {
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
    }
    os.makedirs(f"checkpoints/{model_config.name}", exist_ok=True)
    torch.save(base_checkpoint, f"checkpoints/{model_config.name}/bare_model.pth")
    print(f"New model {model_config.name} built and saved.")
    print(
        f"Number of parameters in model: {sum(p.numel() for p in model.parameters())}"
    )
    del base_checkpoint
    return model


def get_latest_checkpoint(model_name, base_name="checkpoint"):
    model_dir = f"checkpoints/{model_name}/"
    pattern = re.compile(base_name + r"_(\d+)\.pth")

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


def clone_model(model: torch.nn.Module, requires_grad=False) -> torch.nn.Module:
    cloned_model = type(model)(model.config)
    cloned_model.load_state_dict(model.state_dict())
    if not (requires_grad):
        for param in cloned_model.parameters():
            param.requires_grad = requires_grad
    return cloned_model


def load_model(model_name, number=None, special_name=None):
    """
    Loads model for inference.
    Return model, encoder, checkpoint
    """
    if special_name is None:
        if number is None:
            checkpoint_path = get_latest_checkpoint(model_name)
        else:
            checkpoint_path = f"checkpoints/{model_name}/checkpoint_{number}.pth"
    else:
        checkpoint_path = f"checkpoints/{model_name}/{special_name}.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )

    model_config = checkpoint["model_config"]
    model = models.ChessNet(config=model_config)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model from {checkpoint_path}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")
    return model, model_config, checkpoint
