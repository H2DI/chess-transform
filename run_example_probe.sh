#!/bin/bash
# Quick example: Train a linear probe on layer 14
# This uses a small sample of data for a quick test run

echo "=========================================="
echo "Quick Probe Training Example"
echo "=========================================="
echo ""
echo "Training a linear probe on layer 14 with 10K samples..."
echo ""

python train_probe.py \
    --layer 14 \
    --probe-type linear \
    --max-train-samples 10000 \
    --epochs 5 \
    --batch-size 128 \
    --cache-activations

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Check results in: checkpoints/probes/"
echo ""
echo "Next steps:"
echo "  1. Try different layers: --layer 0, 7, 21, 27"
echo "  2. Train with more data: --max-train-samples 100000"
echo "  3. Try MLP probe: --probe-type mlp --hidden-dim 512"
echo "  4. Analyze all layers: python analyze_layers.py"
