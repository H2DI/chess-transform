import os
import pickle
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import chess_seq.models as models
import chess_seq.utils as utils
import chess_seq.training.trainer as trainer
from chess_seq.data import datasets
from chess_seq.evaluation import testing_model
from chess_seq.training.training_config_classes import TrainingSession


class ChessTrainerRunner:
    def __init__(
        self, session_config: TrainingSession, model_config=None, training_config=None
    ):
        self.config = session_config
        self.device = torch.device(self.config.device_str)
        self.model_name = session_config.model_name

        if session_config.new_model:
            self._initialize_new_model(model_config)
            self._initialize_training(training_config)

        self.training_state = self._load_checkpoint()
        self.model_config, self.training_config = self._load_configs()
        self.model = self._setup_model()
        self.optimizer, self.scheduler = self._setup_training()
        self.encoder = self.training_state["encoder"]
        self.writer = SummaryWriter(
            log_dir=f"runs/chess_transformer_experiment/{self.model_config.name}"
        )

        self.data_format = self.config.data_format
        self.n_steps = self.training_state["n_steps"]
        self.n_games = self.training_state["n_games"]
        if self.config.restart:
            self.epoch = 0
            self.file_number = 0
        else:
            self.epoch = self.training_state["epoch"]
            self.file_number = self.training_state["file_number"]

        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.model_config.vocab_size
        )

    def _load_checkpoint(self):
        checkpoint_path = utils.get_latest_checkpoint(self.model_name)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        return checkpoint

    def _load_configs(self):
        model_config = self.training_state["model_config"]
        training_config = self.training_state["training_config"]
        return model_config, training_config

    def _setup_model(self):
        model = models.ChessNet(config=self.model_config).to(self.device)
        model.load_state_dict(self.training_state["model_state_dict"])
        return model

    def _setup_training(self):
        optimizer, scheduler = trainer.initialize_optimizer(
            self.training_config, self.model
        )
        optimizer.load_state_dict(self.training_state["optimizer_state_dict"])
        scheduler.load_state_dict(self.training_state["scheduler_state_dict"])
        return optimizer, scheduler

    def _initialize_new_model(self, model_config):
        assert model_config is not None, (
            "Model configuration must be provided for new models"
        )
        assert model_config.name == self.config.model_name, "Model name mismatch"
        self.model_config = model_config
        with open(model_config.encoder_path, "rb") as f:
            self.encoder = pickle.load(f)
        self.model = models.ChessNet(config=model_config).to(self.device)

    def _initialize_training(self, training_config):
        assert training_config is not None, (
            "Training configuration must be provided for new models"
        )
        self.training_config = training_config
        self.n_steps, self.n_games, self.epoch, self.file_number = 0, 0, 0, 0
        self.optimizer, self.scheduler = trainer.initialize_optimizer(
            training_config, self.model
        )
        self.save_checkpoint()

    def train(self, skip_seen_files=True):
        train_folder = self.config.data_folder
        print(f"{train_folder=}")
        print(f"{self.data_format=}")
        train_files = [
            train_folder + f
            for f in os.listdir(train_folder)
            if f.endswith(f".{self.data_format}")
        ]
        for epoch in range(self.config.num_epochs - self.epoch):
            for file_id, train_file in enumerate(train_files):
                if file_id + 1 < self.file_number and skip_seen_files:
                    print(f"Skipping {train_file} as it is already processed.")
                    continue

                self.file_number += 1
                print(f"File : {train_file}")
                self._train_on_file(train_file)
            self.file_number = 0
            self.epoch += 1

    def _train_on_file(self, train_file):
        dataloader = datasets.build_dataloader(
            train_file,
            batch_size=self.training_config.batch_size,
            device=self.device,
            padding_value=self.model_config.vocab_size,
            data_format=self.config.data_format,
        )

        self._count_lines(train_file)

        self.model.train()
        print("Start training")
        for i, seq in tqdm(enumerate(dataloader)):
            self.n_steps += 1
            self.n_games += self.training_config.batch_size

            loss, logits = self._train_step(seq)

            with torch.no_grad():
                self.writer.add_scalar("Loss/train", loss.item(), self.n_steps)
                self.writer.add_scalar(
                    "LR", self.scheduler.get_last_lr()[0], self.n_steps
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()
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
        if self.data_format == "npz":
            with open(train_file, "rb") as f:
                data = np.load(f)
                num_lines = len(data)
        elif self.data_format == "csv":
            with open(train_file, "r") as f:
                num_lines = sum(1 for _ in f)
        print(f"Number of games in training file: {num_lines}")

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

        b, T = seq.shape

        tgt_mask = torch.tril(torch.ones(T - 1, T - 1)).to(self.device).bool()
        logits = self.model(input_seq, mask=tgt_mask)
        loss = self.criterion(logits.view(-1, logits.size(-1)), target.reshape(-1))
        return loss, logits

    def save_checkpoint(self):
        checkpoint = {
            "model_config": self.model_config,
            "training_config": self.training_config,
            "n_steps": self.n_steps,
            "n_games": self.n_games,
            "epoch": self.epoch,
            "file_number": self.file_number,
            "encoder": self.encoder,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        name = self.model_config.name
        os.makedirs(f"checkpoints/{name}", exist_ok=True)
        torch.save(checkpoint, f"checkpoints/{name}/checkpoint_{self.n_games}.pth")
        print(f"Saved at checkpoint_{self.n_games}.pth")
