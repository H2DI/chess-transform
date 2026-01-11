import os
import re
import sys
import shutil
import torch

from huggingface_hub import hf_hub_download

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from safetensors.torch import load_file

from ..encoder import MoveEncoder
from ..models import ChessNet
from ..configs import ModelConfig
from .. import configs as _configs


# Default Hugging Face Hub repository for chess-transform models
DEFAULT_HF_REPO = "h2di/chess-transform-gamba-rossa"


def build_and_save_model(model_config):
    model = ChessNet(config=model_config)
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
    script_path = os.path.realpath(__file__)
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    model_dir = repo_path + f"/checkpoints/{model_name}/"
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


def load_model_from_safetensors(model_name):
    """
    Loads model from safetensors format using JSON config.
    """
    script_path = os.path.realpath(__file__)
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

    model_dir_path = os.path.join(repo_path, f"checkpoints/{model_name}")

    config_path = os.path.join(model_dir_path, "config.json")
    model_path = os.path.join(model_dir_path, "model.safetensors")
    encoder_path = os.path.join(model_dir_path, "id_to_token.json")

    encoder = MoveEncoder().load(encoder_path)
    model_cfg = ModelConfig.from_json_file(config_path)
    model = ChessNet(config=model_cfg)

    print(f"Loading model weights from {model_path}")
    state_dict = load_file(model_path)

    model.load_state_dict(state_dict)

    print("Model loaded from safetensors.")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")

    return model, model_cfg, encoder


def load_model_from_hf(
    repo_id: str = None,
    model_name: str = "chess-transform-gamba-rossa",
    force_download: bool = False,
):
    """
    Loads the chess model from Hugging Face Hub, caching locally in checkpoints/.

    On first call, downloads model.safetensors, config.json, and id_to_token.json
    from the HF repo and saves them all to checkpoints/{model_name}/.

    On subsequent calls, loads from local files unless force_download=True.

    Args:
        repo_id: The Hugging Face repository ID (e.g., "h2di/chess-transform-gamba-rossa").
                 If None, uses the default repository.
        model_name: Name for local storage folder (default: "gamba_rossa").
        force_download: If True, re-download from HF even if local files exist.

    Returns:
        tuple: (model, model_config, encoder)

    Example:
        >>> model, config, encoder = load_model_from_hf()
        >>> model.eval()
        >>> # Ready to use!
    """
    script_path = os.path.realpath(__file__)
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

    # All files go in checkpoints/{model_name}/
    model_dir = os.path.join(repo_path, f"checkpoints/{model_name}")
    local_config_path = os.path.join(model_dir, "config.json")
    local_model_path = os.path.join(model_dir, "model.safetensors")
    local_encoder_path = os.path.join(model_dir, "id_to_token.json")

    # Check if all local files exist
    all_exist = (
        os.path.exists(local_config_path)
        and os.path.exists(local_model_path)
        and os.path.exists(local_encoder_path)
    )

    if all_exist and not force_download:
        print(f"Loading model from local cache: checkpoints/{model_name}/")
    else:
        # Download from HF Hub
        if repo_id is None:
            repo_id = DEFAULT_HF_REPO

        print(f"Downloading model from Hugging Face Hub: {repo_id}")

        # Download files
        config_hf_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
        )
        model_hf_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
        )
        encoder_hf_path = hf_hub_download(
            repo_id=repo_id,
            filename="id_to_token.json",
        )

        # Copy all to checkpoints/{model_name}/

        os.makedirs(model_dir, exist_ok=True)

        shutil.copy2(config_hf_path, local_config_path)
        shutil.copy2(model_hf_path, local_model_path)
        shutil.copy2(encoder_hf_path, local_encoder_path)

        print(f"Downloaded to checkpoints/{model_name}/")

    model_cfg = ModelConfig.from_json_file(local_config_path)
    encoder = MoveEncoder().load(local_encoder_path)
    model = ChessNet(config=model_cfg)
    print(f"Loading model weights from {local_model_path}")
    state_dict = load_file(local_model_path)
    model.load_state_dict(state_dict)

    print("Model loaded.")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    return model, model_cfg, encoder


def load_model_from_checkpoint(model_name, number=None, special_name=None):
    """
    Loads model from a pickled checkpoint
    """

    # Older checkpoints were pickled with top-level module name "configs".
    # Register an alias to the current chess_seq.configs so torch.load can resolve it.
    sys.modules.setdefault("configs", _configs)

    script_path = os.path.realpath(__file__)
    repo_path = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    print(f"Repository path: {repo_path}")

    if special_name is None:
        if number is None:
            checkpoint_path = get_latest_checkpoint(model_name)
        else:
            checkpoint_path = (
                repo_path + f"/checkpoints/{model_name}/checkpoint_{number}.pth"
            )
    else:
        checkpoint_path = repo_path + f"/checkpoints/{model_name}/{special_name}.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    print(f"Loading model from {checkpoint_path}")

    model_config = checkpoint["model_config"]
    model = ChessNet(config=model_config)
    sd = checkpoint["model_state_dict"]
    consume_prefix_in_state_dict_if_present(sd, "_orig_mod.")

    info = {"n_games": checkpoint.get("n_games", 0)}
    del checkpoint

    model.load_state_dict(sd)

    print("Model loaded.")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {num_params}")
    return model, model_config, info
