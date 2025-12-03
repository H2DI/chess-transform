import os
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import chess_seq.models as models
import chess_seq.utils as utils
import chess_seq.training.trainer as trainer
from chess_seq.data import datasets
from chess_seq.evaluation import testing_model
from chess_seq.encoder import MoveEncoder

from configs import TrainingSession, ModelConfig, TrainingConfig


class ChessTrainerRunner:
    def __init__(
        self,
        session_config: TrainingSession,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
    ):
        self.config = session_config
        self.device = torch.device(self.config.device_str)
        self.model_name = session_config.model_name

        if session_config.resume:
            training_state = self._load_checkpoint()
            self._load_configs(training_state)
            self._setup_model(training_state)
            self._setup_training(training_state)
            del training_state
        else:
            assert model_config is not None, (
                "Model configuration must be provided for new models"
            )
            assert training_config is not None, (
                "Training configuration must be provided for new models"
            )
            self.model_config = model_config
            self.training_config = training_config
            self._initialize_new_model()
            self._initialize_training()
            self.save_checkpoint()

        self.encoder = MoveEncoder()
        self.encoder.load(self.model_config.encoder_path)

        self.writer = SummaryWriter(
            log_dir=f"runs/chess_transformer_experiment/{self.model_config.name}"
        )

        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.model_config.pad_index
        )

    def _load_checkpoint(self):
        checkpoint_path = utils.get_latest_checkpoint(self.model_name)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        return checkpoint

    def _load_configs(self, training_state):
        self.model_config = training_state["model_config"]
        self.training_config = training_state["training_config"]

    def _setup_model(self, training_state):
        model = models.ChessNet(config=self.model_config).to(self.device)
        model.load_state_dict(training_state["model_state_dict"])
        self.model = model
        if self.config.jit:
            self.model = torch.compile(self.model)

    def _setup_training(self, training_state):
        self.optimizer, self.scheduler = trainer.initialize_optimizer(
            self.training_config, self.model
        )
        self.optimizer.load_state_dict(training_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(training_state["scheduler_state_dict"])

        self.n_steps = training_state["n_steps"]
        self.n_games = training_state["n_games"]
        self.epoch = training_state["epoch"]
        self.file_number = training_state["file_number"]

    def _initialize_new_model(self):
        self.model = utils.build_and_save_model(self.model_config)
        self.model = self.model.to(self.device)
        if self.config.jit:
            self.model = torch.compile(self.model)

    def _initialize_training(self):
        self.n_steps, self.n_games, self.epoch, self.file_number = 0, 0, 0, 0
        self.optimizer, self.scheduler = trainer.initialize_optimizer(
            self.training_config, self.model
        )

    def train(self, skip_seen_files=True):
        train_folder = self.config.data_folder
        print(f"{train_folder=}")
        train_files = sorted(
            [train_folder + f for f in os.listdir(train_folder) if f.endswith(".npz")]
        )
        for epoch in range(self.epoch, self.training_config.num_epochs):
            for file_id, train_file in enumerate(train_files):
                if file_id + 1 < self.file_number and skip_seen_files:
                    print(f"Skipping {train_file} as it is already processed.")
                    continue

                self.file_number += 1
                print(f"File : {train_file}")
                self._train_on_file(train_file)
            self.file_number = 0
            self.epoch = epoch + 1
            self.save_checkpoint()

    def _train_on_file(self, train_file):
        dataloader = datasets.build_dataloader(
            train_file,
            batch_size=self.training_config.batch_size,
            padding_value=self.model_config.pad_index,
        )

        self._count_lines(train_file)

        self.model.train()
        print("Start training")
        for i, seq in tqdm(enumerate(dataloader)):
            print(f"Training step {i}, seq shape: {seq.shape}")
            self.n_steps += 1
            self.n_games += seq.size(0)

            self.optimizer.zero_grad()
            loss = self._train_step(seq)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            self.writer.add_scalar(
                "Loss/train", float(loss.detach().cpu()), self.n_steps
            )
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], self.n_steps)
            if i % 100 == 0:
                trainer.log_grads(self.writer, self.model, self.n_steps)
                trainer.log_weight_norms(self.writer, self.model, self.n_steps)

            if i % self.config.test_interval == 0:
                self._evaluate_model()

            if i % self.config.checkpoint_interval == 0 and i > 0:
                self.save_checkpoint()

        self.save_checkpoint()
        self._evaluate_model()

    def _count_lines(self, train_file):
        with open(train_file, "rb") as f:
            data = np.load(f)
            num_lines = len(data["game_ids"])
        print(f"Number of games in training file: {num_lines}")

    @torch.no_grad()
    def _evaluate_model(self):
        self.model.eval()
        testing_model.eval_legal_moves_and_log(
            self.model,
            self.encoder,
            self.writer,
            self.n_games,
            self.config.test_games_lengths,
        )
        self.model.train()

    def _train_step(self, seq):
        seq = seq.to(self.device)

        input_seq = seq[:, :-1]
        target = seq[:, 1:]

        logits = self.model(input_seq)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def save_checkpoint(self):
        checkpoint = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "n_steps": self.n_steps,
            "n_games": self.n_games,
            "epoch": self.epoch,
            "file_number": self.file_number,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        name = self.model_config.name
        os.makedirs(f"checkpoints/{name}", exist_ok=True)
        torch.save(checkpoint, f"checkpoints/{name}/checkpoint_{self.n_games}.pth")
        print(f"Saved at checkpoint_{self.n_games}.pth")
