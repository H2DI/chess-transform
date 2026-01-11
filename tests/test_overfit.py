from dataclasses import dataclass, field
from typing import List

import torch

from chess_seq.configs import ModelConfig, TrainingSession, TrainingConfig
from chess_seq import ChessNet, ChessTrainerRunner, build_dataloader

from tqdm import tqdm


# ROOT = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(ROOT)

# Smoke test the model initialization


@dataclass
class ModelConfigTiny:
    name: str = "new_test"
    vocab_size: int = 4611  # padding_id is last
    block_size: int = 512
    k: int = 128
    head_dim: int = 32
    n_head: int = 8
    n_layers: int = 6
    dropout: int = 0.0
    kv_groups: int = 2
    ff_expansion: int = 3

    pad_index: int = 4610
    special_freqs: List[float] = field(default_factory=lambda: [2])
    encoder_path: str = "data/id_to_token.json"

    device: str = "cpu"


### Test a few training steps ###

print("Testing a few training steps on tiny model")

cfg = ModelConfigTiny()
model = ChessNet(cfg).to("cpu")

npz_path = "data/train_npz/shard_00000.npz"
loader = build_dataloader(npz_path, batch_size=16, shuffle=True)

seq_batch = next(iter(loader))
criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.vocab_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
losses = []

seq = seq_batch.to(cfg.device)
for _ in tqdm(range(50)):
    input_seq = seq[:, :-1]
    target = seq[:, 1:]

    logits = model(input_seq)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        target.reshape(-1),
        reduction="mean",
        ignore_index=cfg.pad_index,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if losses[-1] < 1:
        print("Loss below 1, stopping test.")
print(losses)

#### Test a full model.

cfg = ModelConfig()
model = ChessNet(cfg)

# Smoke test the trainer runner initialization
runner = ChessTrainerRunner(
    session_config=TrainingSession(),
    model_config=cfg,
    training_config=TrainingConfig(),
)

npz_path = "data/train_npz/testing_data_16.npz"
runner._train_on_file(npz_path)

print("ChessTrainerRunner initialized successfully.")
