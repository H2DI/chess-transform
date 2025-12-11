"""
Utilities for extracting activations from ChessNet model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ActivationExtractor:
    """
    Extract intermediate activations from a model using forward hooks.
    """

    def __init__(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Args:
            model: The model to extract activations from
            layer_names: List of layer names to extract from. If None, extracts from all DecoderLayers.
        """
        self.model = model
        self.activations = {}
        self.hooks = []

        if layer_names is None:
            # Auto-detect decoder layers
            layer_names = [f"blocks.{i}" for i in range(len(model.blocks))]

        self.layer_names = layer_names
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on specified layers."""

        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()

            return hook

        for name in self.layer_names:
            # Navigate to the layer using the name
            layer = self._get_layer_by_name(name)
            if layer is not None:
                handle = layer.register_forward_hook(get_activation(name))
                self.hooks.append(handle)

    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a layer by its name path (e.g., 'blocks.0')."""
        parts = name.split(".")
        module = self.model

        try:
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module
        except (AttributeError, IndexError, TypeError):
            print(f"Warning: Could not find layer {name}")
            return None

    def extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run forward pass and extract activations.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to activations
        """
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(x)
        return dict(self.activations)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def __del__(self):
        """Cleanup hooks when extractor is deleted."""
        self.remove_hooks()


class MultiLayerExtractor:
    """
    Extract activations from multiple specific layers at once.
    More efficient than running multiple passes.
    """

    def __init__(self, model: nn.Module, extract_layers: List[int]):
        """
        Args:
            model: The ChessNet model
            extract_layers: List of layer indices to extract from (e.g., [0, 7, 14, 27])
        """
        self.model = model
        self.extract_layers = extract_layers
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on specified layer indices."""

        def get_activation(layer_idx):
            def hook(module, input, output):
                self.activations[layer_idx] = output.detach()

            return hook

        for layer_idx in self.extract_layers:
            if 0 <= layer_idx < len(self.model.blocks):
                handle = self.model.blocks[layer_idx].register_forward_hook(
                    get_activation(layer_idx)
                )
                self.hooks.append(handle)

    def extract(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Run forward pass and extract activations.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer indices to activations
        """
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(x)
        return dict(self.activations)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def __del__(self):
        """Cleanup hooks when extractor is deleted."""
        self.remove_hooks()


class ResidualStreamExtractor:
    """
    Extract residual stream activations (after each block, before final layer norm).
    This is often the most informative representation for probing.
    """

    def __init__(self, model: nn.Module, layer_indices: Optional[List[int]] = None):
        """
        Args:
            model: The ChessNet model
            layer_indices: Which layers to extract from. If None, extracts from all layers.
        """
        self.model = model
        self.activations = []
        self.hooks = []

        if layer_indices is None:
            layer_indices = list(range(len(model.blocks)))

        self.layer_indices = layer_indices
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture residual stream."""

        def get_activation(layer_idx):
            def hook(module, input, output):
                # output is the residual stream after this block
                self.activations.append((layer_idx, output.detach()))

            return hook

        for layer_idx in self.layer_indices:
            if 0 <= layer_idx < len(self.model.blocks):
                handle = self.model.blocks[layer_idx].register_forward_hook(
                    get_activation(layer_idx)
                )
                self.hooks.append(handle)

    def extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run forward pass and extract residual stream activations.

        Args:
            x: Input tensor

        Returns:
            List of activations in order of layer_indices
        """
        self.activations.clear()
        with torch.no_grad():
            _ = self.model(x)

        # Sort by layer index and return just the activations
        self.activations.sort(key=lambda x: x[0])
        return [act for _, act in self.activations]

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()

    def __del__(self):
        """Cleanup hooks when extractor is deleted."""
        self.remove_hooks()


def extract_layer_activations(
    model: nn.Module, x: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    """
    Convenience function to extract activations from a single layer.

    Args:
        model: The ChessNet model
        x: Input tensor
        layer_idx: Index of the layer to extract from

    Returns:
        Activations tensor
    """
    extractor = MultiLayerExtractor(model, [layer_idx])
    activations = extractor.extract(x)
    extractor.remove_hooks()
    return activations[layer_idx]


def extract_all_layers(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract activations from all decoder layers.

    Args:
        model: The ChessNet model
        x: Input tensor

    Returns:
        List of activation tensors, one per layer
    """
    num_layers = len(model.blocks)
    extractor = ResidualStreamExtractor(model, list(range(num_layers)))
    activations = extractor.extract(x)
    extractor.remove_hooks()
    return activations


if __name__ == "__main__":
    import sys

    sys.path.append("/Users/hadiji/Documents/GitHub/chess-transform")
    from chess_seq.chess_seq.models import ChessNet
    from chess_seq.chess_seq.configs import ModelConfig

    # Create a small test model
    config = ModelConfig(
        vocab_size=4611,
        block_size=64,
        k=128,
        head_dim=32,
        n_head=4,
        n_layers=4,
        dropout=0.0,
        kv_groups=2,
    )

    model = ChessNet(config)
    model.eval()

    # Test input
    batch_size = 2
    seq_len = 16
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print("Testing activation extraction...")

    # Test single layer extraction
    print("\n1. Single layer extraction:")
    act = extract_layer_activations(model, x, layer_idx=2)
    print(f"   Layer 2 activation shape: {act.shape}")

    # Test multi-layer extraction
    print("\n2. Multi-layer extraction:")
    extractor = MultiLayerExtractor(model, [0, 2, 3])
    acts = extractor.extract(x)
    for layer_idx, act in acts.items():
        print(f"   Layer {layer_idx} activation shape: {act.shape}")
    extractor.remove_hooks()

    # Test residual stream extraction
    print("\n3. Residual stream extraction:")
    all_acts = extract_all_layers(model, x)
    print(f"   Extracted {len(all_acts)} layers")
    print(f"   Each activation shape: {all_acts[0].shape}")

    print("\nAll tests passed!")
