# Chess Transformers

A PyTorch implementation of Transformer models for predicting chess move sequences. The current version uses imitation learning (next-token prediction) without explicit reward modeling for move quality or validity.

## Model

The architecture is a reimplementation of Qwen 3 0.6B.

- **Parameters**: Approximately 450M (adjusted for vocabulary size).
- **Tokens**: Move format 'FromToPromotion' (e.g., 'e2e4', 'f8g6', 'e7e8pq').
- **Vocabulary**: 4608 move tokens plus start, end, and pad tokens.

## Data

Training data is sourced from the [Lichess Elite Database](https://database.nikonoel.fr/).

- **Total Games**: 12,421,396
- **Total Tokens**: 1,102,678,752

## Results

The model was trained on an NVIDIA A100 GPU. Illegal moves are masked during inference.

- View sample games played by the model against itself [here](https://lichess.org/study/ZbXAbPvL).
- Play against the bot on Lichess: [GambaRossa](https://lichess.org/@/GambaRossa/all).

## Future Work

- Experiments: length generalization, extracting the chess world model.


- Fine-tuning on high-quality games
- Self-play reinforcement learning. GRPO is implemented but needs refinement.



