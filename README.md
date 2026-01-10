# Chess Transformers

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch implementation of a decoder-only Transformer for chess move prediction. The model learns to play chess through imitation learning on grandmaster games, predicting move sequences as tokens.

<p align="center">
  <a href="https://lichess.org/@/GambaRossa/all">🎮 Play against the bot on Lichess</a> •
  <a href="https://lichess.org/study/ZbXAbPvL">♟️ View sample games</a>
</p>

## Features

- **450M parameter Transformer** based on the Qwen architecture
- **Custom move tokenization**: FromToPromotion format (e.g., `e2e4`, `e7e8pq`)
- **Rotary Position Embeddings (RoPE)** for sequence modeling
- **Group Query Attention** for efficient inference
- **Linear probing tools** for interpretability research

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-transform.git
cd chess-transform

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

### Playing Games with a Trained Model

```python
from chess_seq import load_model, MoveEncoder, ChessGameEngine
import torch

# Load the model
model, config, info = load_model("gamba_rossa", special_name="final")
model.eval()

# Initialize encoder and game engine
encoder = MoveEncoder()
encoder.build()

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

## Results

The model achieves strong play when illegal moves are masked during inference:

- **Self-play**: Generates coherent full games
- **Live testing**: Available on [Lichess as GambaRossa](https://lichess.org/@/GambaRossa/all)
- **Sample games**: [Lichess Study](https://lichess.org/study/ZbXAbPvL)

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



