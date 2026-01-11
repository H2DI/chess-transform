import torch
import pytest
from chess_seq import (
    load_model_from_checkpoint,
    load_model_from_safetensors,
)


def test_load_model_gamba_rossa():
    """Test loading the gamba_rossa model from checkpoint"""
    model, model_config, info = load_model_from_checkpoint("gamba_rossa")

    assert model is not None
    assert model_config is not None
    assert isinstance(model, torch.nn.Module)
    assert "n_games" in info

    # Verify model has parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0


def test_load_model_gamba_rossa_safetensors():
    """Test loading the gamba_rossa model from safetensors format"""
    model, model_config, encoder = load_model_from_safetensors("gamba_rossa")

    assert model is not None
    assert model_config is not None
    assert encoder is not None
    assert isinstance(model, torch.nn.Module)

    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0


def test_model_forward_pass():
    """Test that loaded model can perform forward pass"""
    model, model_config, _ = load_model_from_checkpoint("gamba_rossa")
    model.eval()

    # Create dummy input (adjust shape based on your model's expected input)
    batch_size = 2
    seq_length = 10
    dummy_input = torch.randint(0, model_config.vocab_size, (batch_size, seq_length))

    with torch.no_grad():
        output = model(dummy_input)

    assert output is not None
    assert output.shape[0] == batch_size
