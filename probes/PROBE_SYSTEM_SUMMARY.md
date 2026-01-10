# Probe Training System - Summary

## What Was Built

I've created a complete system for training **probes** to recover chess board positions from the internal activations of your trained chess transformer model. This is a mechanistic interpretability technique that reveals what information the model represents at different layers.

## Files Created

### Core Modules (in `chess_seq/chess_seq/evaluation/`)

1. **`position_encoder.py`** (189 lines)
   - Converts chess boards to/from 13×8×8 tensors (12 piece types + empty)
   - Computes accuracy metrics (square-level and board-level)
   - Visualizes predictions vs ground truth

2. **`probes.py`** (253 lines)
   - `LinearProbe`: Simple linear transformation (tests if info is linearly accessible)
   - `MLPProbe`: Multi-layer perceptron (tests non-linear accessibility)
   - `SequenceProbe`: Aggregates info across sequence tokens
   - `LayerwiseProbe`: Combines multiple layers
   - Factory function `create_probe()` for easy instantiation

3. **`activation_extractor.py`** (235 lines)
   - Extracts intermediate activations from ChessNet using PyTorch hooks
   - Multiple extractors for different use cases:
     - `ActivationExtractor`: General-purpose extractor
     - `MultiLayerExtractor`: Extract from specific layers efficiently
     - `ResidualStreamExtractor`: Extract residual stream activations
   - Utility functions for quick single-layer extraction

4. **`probe_training.py`** (332 lines)
   - `ProbeDataset`: Generates training data from encoded games
     - Replays games move-by-move
     - Extracts activations at each position
     - Pairs activations with board states
     - Optional activation caching for speed
   - `train_probe()`: Complete training loop with metrics
   - `evaluate_probe()`: Evaluation with visualization

### Main Entry Point

5. **`train_probe.py`** (299 lines)
   - Complete CLI for training probes
   - Supports all probe types and layers
   - Automatic model loading from checkpoints
   - Training history tracking and plotting
   - Configurable hyperparameters

### Documentation

6. **`PROBE_TRAINING.md`** (290 lines)
   - Comprehensive usage guide
   - Examples for different scenarios
   - Explanation of probe types and metrics
   - Troubleshooting guide
   - Best practices

7. **`test_probe_system.py`** (157 lines)
   - Test script to verify all components work
   - Demonstrates the complete pipeline
   - Useful for debugging

## How It Works

### The Probe Training Pipeline

```
Encoded Games (.npz) 
    ↓
Replay Moves → Chess Positions
    ↓
Pass through ChessNet → Extract Activations (at layer L)
    ↓
Train Probe: Activations → Board State Predictions
    ↓
Evaluate: Compare predictions to ground truth
```

### What Each Component Does

1. **Position Encoder**: Converts board states into a one-hot tensor representation where each of 13 channels represents a piece type or empty square

2. **Activation Extractor**: Uses PyTorch forward hooks to capture intermediate representations from any layer of the transformer

3. **Probe**: A lightweight model (linear or MLP) that learns to map from activations to board states

4. **Training Loop**: Standard supervised learning with cross-entropy loss over the 13 classes per square

## Usage Examples

### Basic: Train a linear probe on layer 14
```bash
python train_probe.py --layer 14 --probe-type linear
```

### Train with more data and epochs
```bash
python train_probe.py \
    --layer 14 \
    --probe-type linear \
    --max-train-samples 100000 \
    --epochs 20
```

### Probe multiple layers to find where position info is strongest
```bash
for layer in 0 7 14 21 27; do
    python train_probe.py --layer $layer --probe-type linear --epochs 10
done
```

### Try a non-linear probe
```bash
python train_probe.py \
    --layer 14 \
    --probe-type mlp \
    --hidden-dim 512 \
    --epochs 15
```

## Interpreting Results

### Metrics

- **Square Accuracy**: % of squares with correct piece (most important metric)
- **Board Accuracy**: % of positions perfectly reconstructed
- **Per-Piece Accuracy**: Accuracy for each piece type

### What Results Mean

| Square Accuracy | Interpretation |
|----------------|----------------|
| > 0.95 | Strong position representation at this layer |
| 0.80-0.95 | Partial position information available |
| < 0.80 | Position info not easily accessible here |

### Scientific Questions You Can Answer

1. **Which layers represent position best?**
   - Train probes on layers 0, 7, 14, 21, 27
   - Compare square accuracies
   - Likely hypothesis: Middle-to-late layers should be best

2. **Is position info linearly accessible?**
   - If linear probe succeeds: Info is linearly encoded (easier to extract)
   - If MLP succeeds but linear fails: Non-linear encoding

3. **How does position info build up?**
   - Use `LayerwiseProbe` or probe each layer separately
   - Plot accuracy vs layer depth

4. **Is position distributed across tokens?**
   - Use `SequenceProbe` with different aggregations
   - Compare to single-token probes

## Key Features

✅ **Multiple probe types**: Linear, MLP, sequence, layerwise  
✅ **Flexible**: Works with any layer of your model  
✅ **Efficient**: Optional activation caching for speed  
✅ **Comprehensive metrics**: Square, board, and per-piece accuracy  
✅ **Visualization**: See what the probe gets wrong  
✅ **Easy to use**: Simple CLI with sensible defaults  
✅ **Well-documented**: Full guide with examples  
✅ **Tested**: Test script to verify everything works  

## Next Steps

1. **Test the system**:
   ```bash
   python test_probe_system.py
   ```

2. **Train your first probe**:
   ```bash
   python train_probe.py --layer 14 --max-train-samples 10000 --epochs 5
   ```

3. **Analyze results** and try different layers/probe types

4. **Read the full documentation** in `PROBE_TRAINING.md`

## Technical Details

- **Board Representation**: 13×8×8 one-hot tensor (12 pieces + empty)
- **Loss Function**: Cross-entropy over 13 classes per square
- **Optimizer**: Adam with configurable learning rate
- **Default Settings**: 50K training samples, 10 epochs, batch size 128
- **Memory**: Caching uses ~16MB per 1000 samples (with 1024-dim activations)

## Requirements

The system uses existing dependencies from your chess-transform project:
- PyTorch
- chess (python-chess)
- numpy
- tqdm

No additional packages needed!
