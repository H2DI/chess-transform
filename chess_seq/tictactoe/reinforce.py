import os

import numpy as np
import torch
import torch.optim as optim

import chess_seq.tictactoe.mechanics as mechanics
from chess_seq.tictactoe.agent import TTTAgent
import chess_seq.utils as utils


class REINFORCE(TTTAgent):
    def __init__(
        self,
        model,
        encoder,
        base_name,
        gamma,
        episode_batch_size,
        learning_rate,
    ):
        super().__init__(model, encoder)
        self.full_name = base_name + "_reinforce"

        self.gamma = gamma

        self.episode_batch_size = episode_batch_size
        self.learning_rate = learning_rate

        self.reset()

    def reset(self):
        self.scores = []
        self.current_episode = []
        self.current_episode_tokenids_played = []

        self.optimizer = optim.Adam(
            params=self.engine.model.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0

    def new_game(self):
        pass

    def update(self, state, action, reward, done, next_state, writer=None):
        """ """
        tokenid = np.array([self.last_tokenid])

        self.current_episode.append(
            (
                torch.tensor(tokenid, dtype=torch.int64),
                torch.tensor([reward]),
            )
        )

        if done:
            self.n_eps += 1

            tokenids, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            current_episode_returns = self._gradient_returns(rewards, self.gamma)
            current_episode_returns = (
                current_episode_returns - current_episode_returns.mean()
            ) / (current_episode_returns.std() + 1e-3)

            sequence = mechanics.move_stack_to_sequence(
                state, self.engine.encoder, self.engine.device
            )
            unn_log_probs = self.engine.model(sequence)
            log_probs = torch.log_softmax(unn_log_probs, dim=-1)

            if len(state) % 2 == 0:
                log_probs = log_probs[:, ::2, :]
            elif len(state) % 2 == 1:
                log_probs = log_probs[:, 1::2, :]

            tokenids = tokenids.transpose(0, 1)
            self.scores.append(
                torch.dot(
                    log_probs.gather(-1, tokenids).squeeze(0).squeeze(1),
                    current_episode_returns,
                ).unsqueeze(0)
            )
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size) == 0:
                self.optimizer.zero_grad()
                full_neg_score = -torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()
                self.optimizer.step()

                self.scores = []

                if writer is not None:
                    total_norm = 0.0
                    for p in self.engine.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    writer.add_scalar("train/grad_norm", total_norm, self.n_eps)
                    # print(f"Gradient norm: {total_norm:.4f}")

        return

    def _gradient_returns(self, rewards, gamma):
        """
        Turns a list of rewards into the list of returns * gamma**t
        """
        G = 0
        returns_list = []
        T = len(rewards)
        full_gamma = np.power(gamma, T)
        for t in range(T):
            G = rewards[T - t - 1] + gamma * G
            full_gamma /= gamma
            returns_list.append(full_gamma * G)
        return torch.tensor(returns_list[::-1])  # , dtype=torch.float32)

    def load_rl_checkpoint(self):
        checkpoint_path = utils.get_latest_checkpoint(self.full_name)

        rl_checkpoint = torch.load(
            checkpoint_path,
            map_location=self.engine.device,
            weights_only=False,
        )

        self.gamma = rl_checkpoint["gamma"]
        self.lr = rl_checkpoint["lr"]
        # self.episode_batch_size = rl_checkpoint["episode_batch_size"]

        self.n_eps = rl_checkpoint["n_eps"]

        self.engine.model.load_state_dict(rl_checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(rl_checkpoint["optimizer_state_dict"])

    def save_rl_checkpoint(self, model_config, checkpoint_name=None):
        rl_checkpoint = {
            "model_config": model_config,
            "encoder": self.engine.encoder,
            "gamma": self.gamma,
            "lr": self.learning_rate,
            # "episode_batch_size": self.episode_batch_size,
            "n_eps": self.n_eps,
            "model_state_dict": self.engine.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        os.makedirs(f"checkpoints/{self.full_name}", exist_ok=True)
        if checkpoint_name is not None:
            full_path = f"checkpoints/{self.full_name}/{checkpoint_name}.pth"
        else:
            full_path = f"checkpoints/{self.full_name}/checkpoint_{self.n_eps}.pth"

        torch.save(rl_checkpoint, full_path)
        print(f"Saved at {full_path}")
