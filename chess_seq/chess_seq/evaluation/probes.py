"""
Probe models for decoding board positions from transformer activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """
    Simple linear probe to decode board state from activations.
    This is the standard approach for linear probing experiments.
    """

    def __init__(self, input_dim: int, output_dim: int = 832):
        """
        Args:
            input_dim: Dimension of the input activations (e.g., model hidden dim)
            output_dim: Dimension of the output (13 channels * 8 * 8 = 832)
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Activations of shape (B, T, D) or (B, D)

        Returns:
            Board predictions of shape (B, 13, 8, 8)
        """
        # If we have sequence dimension, take the last token
        if x.dim() == 3:
            x = x[:, -1, :]  # (B, D)

        out = self.linear(x)  # (B, 832)
        out = out.view(-1, 13, 8, 8)  # (B, 13, 8, 8)
        return out


class MLPProbe(nn.Module):
    """
    MLP probe with one or more hidden layers.
    Can learn more complex non-linear relationships.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 832,
    ):
        """
        Args:
            input_dim: Dimension of the input activations
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout probability
            output_dim: Dimension of the output (13 * 8 * 8 = 832)
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Activations of shape (B, T, D) or (B, D)

        Returns:
            Board predictions of shape (B, 13, 8, 8)
        """
        # If we have sequence dimension, take the last token
        if x.dim() == 3:
            x = x[:, -1, :]  # (B, D)

        out = self.net(x)  # (B, 832)
        out = out.view(-1, 13, 8, 8)  # (B, 13, 8, 8)
        return out


class SequenceProbe(nn.Module):
    """
    Probe that processes the entire sequence of activations.
    Uses attention or pooling to aggregate information across the sequence.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        aggregation: str = "attention",
        output_dim: int = 832,
    ):
        """
        Args:
            input_dim: Dimension of the input activations
            hidden_dim: Dimension of hidden layer
            aggregation: How to aggregate sequence ('attention', 'mean', 'max', 'last')
            output_dim: Dimension of the output (13 * 8 * 8 = 832)
        """
        super().__init__()
        self.aggregation = aggregation

        if aggregation == "attention":
            self.attention = nn.Linear(input_dim, 1)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Activations of shape (B, T, D)

        Returns:
            Board predictions of shape (B, 13, 8, 8)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        # Aggregate sequence information
        if self.aggregation == "last":
            aggregated = x[:, -1, :]  # (B, D)
        elif self.aggregation == "mean":
            aggregated = x.mean(dim=1)  # (B, D)
        elif self.aggregation == "max":
            aggregated = x.max(dim=1)[0]  # (B, D)
        elif self.aggregation == "attention":
            # Compute attention weights
            attn_weights = F.softmax(self.attention(x), dim=1)  # (B, T, 1)
            aggregated = (x * attn_weights).sum(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        out = self.mlp(aggregated)  # (B, 832)
        out = out.view(-1, 13, 8, 8)  # (B, 13, 8, 8)
        return out


class LayerwiseProbe(nn.Module):
    """
    Probe that combines information from multiple layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        hidden_dim: int = 512,
        output_dim: int = 832,
    ):
        """
        Args:
            input_dim: Dimension of activations from each layer
            num_layers: Number of layers to combine
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of the output (13 * 8 * 8 = 832)
        """
        super().__init__()

        # Project each layer to a common dimension
        self.layer_projections = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Combine and decode
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, layer_activations: list) -> torch.Tensor:
        """
        Args:
            layer_activations: List of tensors of shape (B, T, D) or (B, D)

        Returns:
            Board predictions of shape (B, 13, 8, 8)
        """
        # Project and extract last token from each layer
        projected = []
        for i, acts in enumerate(layer_activations):
            if acts.dim() == 3:
                acts = acts[:, -1, :]  # Take last token
            projected.append(self.layer_projections[i](acts))

        # Concatenate all layers
        combined = torch.cat(projected, dim=-1)  # (B, hidden_dim * num_layers)

        out = self.decoder(combined)  # (B, 832)
        out = out.view(-1, 13, 8, 8)  # (B, 13, 8, 8)
        return out


def create_probe(probe_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create different types of probes.

    Args:
        probe_type: Type of probe ('linear', 'mlp', 'sequence', 'layerwise')
        input_dim: Dimension of input activations
        **kwargs: Additional arguments for specific probe types

    Returns:
        Probe model
    """
    if probe_type == "linear":
        return LinearProbe(input_dim)
    elif probe_type == "mlp":
        return MLPProbe(input_dim, **kwargs)
    elif probe_type == "sequence":
        return SequenceProbe(input_dim, **kwargs)
    elif probe_type == "layerwise":
        return LayerwiseProbe(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")


if __name__ == "__main__":
    # Test probes
    batch_size = 4
    seq_len = 16
    hidden_dim = 1024

    # Test linear probe
    probe = LinearProbe(hidden_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = probe(x)
    print(f"LinearProbe output shape: {out.shape}")
    assert out.shape == (batch_size, 13, 8, 8)

    # Test MLP probe
    probe = MLPProbe(hidden_dim)
    out = probe(x)
    print(f"MLPProbe output shape: {out.shape}")
    assert out.shape == (batch_size, 13, 8, 8)

    # Test sequence probe
    probe = SequenceProbe(hidden_dim, aggregation="attention")
    out = probe(x)
    print(f"SequenceProbe output shape: {out.shape}")
    assert out.shape == (batch_size, 13, 8, 8)

    # Test layerwise probe
    num_layers = 4
    layer_acts = [
        torch.randn(batch_size, seq_len, hidden_dim) for _ in range(num_layers)
    ]
    probe = LayerwiseProbe(hidden_dim, num_layers)
    out = probe(layer_acts)
    print(f"LayerwiseProbe output shape: {out.shape}")
    assert out.shape == (batch_size, 13, 8, 8)

    print("All probe tests passed!")
