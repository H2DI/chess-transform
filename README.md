# Chess Transformers

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/h2di/chess-transform-gamba-rossa)

A PyTorch implementation of a decoder-only Transformer for chess move prediction. The model learns to play chess through imitation learning on master games, predicting move sequences as tokens.

<p align="center">
  <a href="https://lichess.org/@/GambaRossa/all">🎮 Play against the bot on Lichess</a> •
  <a href="https://lichess.org/study/ZbXAbPvL">♟️ View sample games</a>
</p>

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-transform.git
cd chess-transform

# Create a conda environment and activate it
conda create -n myenv python=3.9 -y
conda activate myenv


# Install the package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

### Playing Games with a Trained Model

```python
from chess_seq import load_model_from_hf, MoveEncoder, ChessGameEngine
import torch

# Load the model from Hugging Face Hub (auto-downloads on first run)
model, config, encoder = load_model_from_hf()
model.eval()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

engine = ChessGameEngine(model, encoder, device=device)

# Play a game (model vs itself)
game, pgn, bad_plies = engine.play_game(n_plies=80, mask_illegal=True)
print(pgn)
```

### Training a New Model

```python
from chess_seq import ChessTrainerRunner, ModelConfig
from chess_seq.configs import TrainingConfig, TrainingSession

runner = ChessTrainerRunner(
    session_config=TrainingSession(model_name="my_model", device_str="cuda"),
    model_config=ModelConfig(name="my_model"),
    training_config=TrainingConfig(),
)
runner.train()
```

## Model Architecture

| Component | Specification |
|-----------|---------------|
| Parameters | ~450M |
| Layers | 28 |
| Hidden Dim | 1024 |
| Attention Heads | 16 |
| Head Dim | 128 |
| KV Groups | 2 (GQA) |
| Vocabulary | 4,611 tokens |
| Max Sequence | 256 moves |

## Dataset

Training data sourced from the [Lichess Elite Database](https://database.nikonoel.fr/):

| Metric | Value |
|--------|-------|
| Total Games | 12,421,396 |
| Total Tokens | 1,102,678,752 |
| Game Format | PGN |

## Project Structure

```
chess-transform/
├── chess_seq/              # Main package
│   ├── models.py           # Transformer architecture
│   ├── encoder.py          # Move tokenization
│   ├── game_engine.py      # Game playing logic
│   ├── datasets/           # Data loading
│   ├── training/           # Training utilities
│   └── evaluation/         # Probing & analysis
├── probes/                 # Probe training scripts
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
└── train.py                # Training entry point
```

### Reproducing the Evaluation

1. **Download the model from Hugging Face:**

```python
from chess_seq import load_model_from_hf
model, config, encoder = load_model_from_hf()  # Downloads to checkpoints/gamba_rossa/
```

2. **Download Magnus Carlsen games and preprocess:**

```bash
python scripts/download_games.py
```

This downloads games from [pgnmentor.com](https://www.pgnmentor.com/players/Carlsen.zip) and saves tokenized data to `data/carlsen_games/`.

3. **Run the evaluation:**

```bash
python tools/evaluate_model.py \
    --data data/carlsen_games/shard_00000.npz \
    --output reports/carlsen_evaluation.json
```

4. **Generate a Markdown report with plots:**

```bash
python tools/generate_report_markdown.py \
    --input reports/carlsen_evaluation.json \
    --output reports/carlsen_report.md
```

The generated report includes accuracy breakdowns by ply range and side (White/Black).

## Evaluation

We provide tools to evaluate the model on held-out games. Below are results on a set of 640 Magnus Carlsen games:

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 51.49% |
| Top-5 Accuracy | 88.32% |
| Perplexity | 4.58 |
| Legality Rate | 99.87% |

## Interpretability

This project includes tools for probing the model's internal representations:

```python
from chess_seq.evaluation.activation_extractor import ActivationExtractor
from chess_seq.evaluation.probes import LinearProbe

# Extract activations from specific layers
extractor = ActivationExtractor(model, layers=[10, 14, 18])
activations = extractor.extract(sequence)

# Train probes to decode board state
probe = LinearProbe(input_dim=1024, output_dim=768)
```

See [probes/PROBE_QUICKSTART.md](probes/PROBE_QUICKSTART.md) for detailed probing instructions.


## Future Work

- [ ] Length generalization experiments
- [ ] World model extraction and analysis
- [ ] Fine-tuning on high-Elo games
- [ ] Self-play reinforcement learning (GRPO implemented)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{chess_transformers,
  title = {Chess Transformers},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/chess-transform}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Training data from [Lichess Elite Database](https://database.nikonoel.fr/)
- Architecture inspired by [Qwen](https://github.com/QwenLM/Qwen)



