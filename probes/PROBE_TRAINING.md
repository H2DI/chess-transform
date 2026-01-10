# Chess Position Probe Training

This system trains **probes** to recover chess board positions from the internal activations of a trained chess transformer model. This is a mechanistic interpretability technique that helps us understand what information the model represents at different layers.

## Overview

The probe training system consists of several components:

1. **Position Encoder** (`position_encoder.py`): Converts chess board states to/from tensor representations
2. **Probe Models** (`probes.py`): Various probe architectures (linear, MLP, sequence, layerwise)
3. **Activation Extractor** (`activation_extractor.py`): Extracts internal activations from the model
4. **Training Pipeline** (`probe_training.py`): Dataset and training utilities
5. **Main Script** (`train_probe.py`): Entry point for training probes

## Quick Start

### Basic Usage

Train a linear probe on layer 14:

```bash
python train_probe.py --layer 14 --probe-type linear
```

### Train with More Data

```bash
python train_probe.py \
    --layer 14 \
    --probe-type linear \
    --max-train-samples 100000 \
    --epochs 20
```

### Train MLP Probe

```bash
python train_probe.py \
    --layer 27 \
    --probe-type mlp \
    --hidden-dim 512 \
    --epochs 15
```

### Use Multiple Layers

```bash
python train_probe.py \
    --probe-type layerwise \
    --all-layers \
    --hidden-dim 512
```

### Evaluate Only

```bash
python train_probe.py \
    --layer 14 \
    --probe-type linear \
    --evaluate-only
```

## Command Line Arguments

### Model Configuration
- `--model-name`: Name of trained model (default: `gamba_gialla`)
- `--checkpoint`: Specific checkpoint path (default: latest)

### Probe Configuration
- `--probe-type`: Type of probe - `linear`, `mlp`, `sequence`, `layerwise` (default: `linear`)
- `--layer`: Which layer to probe (default: 14)
- `--all-layers`: Use all layers (for layerwise probe)
- `--hidden-dim`: Hidden dimension for MLP probes (default: 512)
- `--aggregation`: Aggregation for sequence probe - `attention`, `mean`, `max`, `last` (default: `attention`)

### Data Configuration
- `--train-data`: Training data file (default: `data/train_npz/train_000.npz`)
- `--val-data`: Validation data file (optional)
- `--max-train-samples`: Max training samples (default: 50000)
- `--max-val-samples`: Max validation samples (default: 5000)
- `--cache-activations`: Cache activations in memory (default: True)

### Training Configuration
- `--epochs`: Number of epochs (default: 10)
- `--batch-size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device - `cuda` or `cpu` (default: auto-detect)

### Output Configuration
- `--output-dir`: Output directory (default: `checkpoints/probes`)
- `--evaluate-only`: Only evaluate existing probe

## Probe Types

### Linear Probe
The simplest probe - a single linear layer mapping activations to board states.
- **Use case**: Test if information is linearly accessible
- **Parameters**: Very few (~800K for 1024-dim model)
- **Speed**: Very fast

```bash
python train_probe.py --probe-type linear --layer 14
```

### MLP Probe
Multi-layer perceptron with configurable hidden layers.
- **Use case**: Test if information is accessible via non-linear transform
- **Parameters**: More parameters, depends on hidden_dim
- **Speed**: Moderate

```bash
python train_probe.py --probe-type mlp --hidden-dim 512 --layer 14
```

### Sequence Probe
Aggregates information across the entire sequence using attention or pooling.
- **Use case**: Test if position info is distributed across tokens
- **Parameters**: Similar to MLP
- **Speed**: Moderate

```bash
python train_probe.py --probe-type sequence --aggregation attention --layer 14
```

### Layerwise Probe
Combines activations from multiple layers.
- **Use case**: Test how information builds up across layers
- **Parameters**: Many parameters (combines all layers)
- **Speed**: Slower

```bash
python train_probe.py --probe-type layerwise --all-layers
```

## Understanding Results

### Metrics

1. **Square Accuracy**: Fraction of squares with correct piece/empty prediction (0-1)
2. **Board Accuracy**: Fraction of boards that are perfectly reconstructed (0-1)
3. **Per-Piece Accuracy**: Accuracy for each piece type separately

### What Good Results Look Like

- **Square Accuracy > 0.95**: Model has strong position representation
- **Square Accuracy 0.80-0.95**: Model has partial position information
- **Square Accuracy < 0.80**: Position info may not be easily accessible at this layer
- **Board Accuracy > 0.50**: Very impressive (perfect reconstruction)

### Interpreting Results

- **Linear probe succeeds**: Information is linearly accessible (easier to extract)
- **MLP probe succeeds, linear fails**: Information is present but non-linearly encoded
- **Both fail**: Information may not be strongly represented at this layer

## Layer-by-Layer Analysis

To understand which layers contain position information, train probes on multiple layers:

```bash
# Early layer
python train_probe.py --layer 2 --probe-type linear

# Middle layers
python train_probe.py --layer 14 --probe-type linear

# Late layers
python train_probe.py --layer 27 --probe-type linear
```

Compare the square accuracies to see where position information is strongest.

## Example Workflow

1. **Train probes on multiple layers**:
   ```bash
   for layer in 0 7 14 21 27; do
       python train_probe.py --layer $layer --probe-type linear --epochs 15
   done
   ```

2. **Compare results** to find which layer(s) best represent position

3. **Try non-linear probe** on best layer:
   ```bash
   python train_probe.py --layer 14 --probe-type mlp --hidden-dim 512 --epochs 20
   ```

4. **Evaluate and visualize**:
   ```bash
   python train_probe.py --layer 14 --probe-type linear --evaluate-only
   ```

## Files and Directory Structure

```
chess_seq/chess_seq/evaluation/
├── position_encoder.py       # Board state encoding/decoding
├── probes.py                  # Probe model architectures
├── activation_extractor.py    # Extract model activations
└── probe_training.py          # Dataset and training loop

train_probe.py                 # Main entry point
checkpoints/probes/            # Saved probe models (created automatically)
```

## Tips and Best Practices

1. **Start with linear probes**: They're fast and tell you if information is easily accessible
2. **Cache activations**: Speeds up training significantly (but uses more RAM)
3. **Use validation data**: Prevents overfitting, especially for MLP probes
4. **Try multiple layers**: Position information may be strongest at specific depths
5. **Monitor board accuracy**: More interpretable than square accuracy
6. **Visualize predictions**: Helps understand what the probe gets wrong

## Advanced Usage

### Custom Data Files

```bash
python train_probe.py \
    --train-data data/train_npz/train_005.npz \
    --val-data data/train_npz/train_006.npz \
    --max-train-samples 200000
```

### Different Model

```bash
python train_probe.py \
    --model-name gamba_rossa \
    --checkpoint checkpoints/gamba_rossa/final.pth
```

### GPU Training

```bash
python train_probe.py \
    --device cuda \
    --batch-size 256 \
    --max-train-samples 500000
```

## Troubleshooting

### Out of Memory
- Reduce `--max-train-samples`
- Disable `--cache-activations`
- Reduce `--batch-size`

### Low Accuracy
- Try different layers
- Increase `--epochs`
- Try MLP probe instead of linear
- Use more training data

### Slow Training
- Enable `--cache-activations`
- Increase `--batch-size` if you have memory
- Use GPU with `--device cuda`

## Citation

This implements standard linear probing techniques commonly used in mechanistic interpretability:
- Alain & Bengio (2016): Understanding intermediate layers using linear classifier probes
- Hewitt & Manning (2019): A Structural Probe for Finding Syntax in Word Representations
