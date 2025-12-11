# Quick Reference: Probe Training

## 🚀 Quick Start

```bash
# Test the system
python test_probe_system.py

# Train your first probe
python train_probe.py --layer 14 --max-train-samples 10000 --epochs 5
```

## 📋 Common Commands

### Train Linear Probe (Recommended Starting Point)
```bash
python train_probe.py --layer 14 --probe-type linear
```

### Train MLP Probe (Non-linear)
```bash
python train_probe.py --layer 14 --probe-type mlp --hidden-dim 512
```

### Analyze All Layers
```bash
python analyze_layers.py --probe-type linear --layers 0 7 14 21 27
```

### Train with More Data
```bash
python train_probe.py --layer 14 --max-train-samples 100000 --epochs 20
```

### Use GPU
```bash
python train_probe.py --layer 14 --device cuda --batch-size 256
```

### Evaluate Existing Probe
```bash
python train_probe.py --layer 14 --evaluate-only
```

## 📊 Key Metrics

| Metric | Good | Okay | Poor |
|--------|------|------|------|
| Square Accuracy | >0.95 | 0.80-0.95 | <0.80 |
| Board Accuracy | >0.50 | 0.20-0.50 | <0.20 |

## 🎯 Probe Types

| Type | Use Case | Speed | Parameters |
|------|----------|-------|------------|
| **linear** | Test linear accessibility | Fast | ~800K |
| **mlp** | Test non-linear patterns | Medium | ~1.5M |
| **sequence** | Aggregate across tokens | Medium | ~1.5M |
| **layerwise** | Combine multiple layers | Slow | Many |

## 🔧 Common Options

```bash
--layer N              # Probe layer N (0 to n_layers-1)
--probe-type TYPE      # linear, mlp, sequence, layerwise
--max-train-samples N  # Use N training samples
--epochs N             # Train for N epochs
--batch-size N         # Batch size (default 128)
--lr RATE              # Learning rate (default 1e-3)
--device DEVICE        # cpu or cuda
--cache-activations    # Cache activations (faster, more RAM)
```

## 📁 Files

| File | Purpose |
|------|---------|
| `train_probe.py` | Main training script |
| `test_probe_system.py` | Test/demo script |
| `analyze_layers.py` | Multi-layer analysis |
| `PROBE_TRAINING.md` | Full documentation |
| `checkpoints/probes/` | Saved probes |
| `reports/layer_analysis/` | Analysis results |

## 🧪 Typical Workflow

1. **Test system**: `python test_probe_system.py`
2. **Quick probe**: `python train_probe.py --layer 14 --max-train-samples 10000 --epochs 5`
3. **Analyze layers**: `python analyze_layers.py --probe-type linear --layers 0 7 14 21 27`
4. **Deep dive**: Train full probe on best layer with more data
5. **Try MLP**: See if non-linear probe improves results

## 💡 Tips

- **Start with linear probes** - they're fast and informative
- **Use --cache-activations** for speed (if you have RAM)
- **Try multiple layers** to find where position info lives
- **Board accuracy** is more interpretable than square accuracy
- **Watch for overfitting** - use validation data for MLP probes

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce `--max-train-samples`
- Remove `--cache-activations`
- Decrease `--batch-size`

**Low Accuracy?**
- Try different `--layer`
- Increase `--epochs`
- Use `--probe-type mlp`
- Increase training data

**Too Slow?**
- Use `--cache-activations`
- Use GPU: `--device cuda`
- Decrease `--max-train-samples`

## 📖 Learn More

- Full guide: `PROBE_TRAINING.md`
- System overview: `PROBE_SYSTEM_SUMMARY.md`
- Code: `chess_seq/chess_seq/evaluation/`
