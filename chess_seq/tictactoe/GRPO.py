import os

import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence

import chess_seq.tictactoe.mechanics as mechanics
from chess_seq.tictactoe.agent import TTTAgent
import chess_seq.utils as utils


class GRPO(TTTAgent):
    def __init__(
        self,
        model,
        encoder,
        base_name,
        beta,
        epsilon_low,
        epsilon_high,
        group_size,
        n_groups,
        learning_rate,
    ):
        self.full_name = base_name + "_GRPO"
        super().__init__(model, encoder, full_name=self.full_name)
        self.engine.model.eval()
        for p in self.engine.model.parameters():
            p.requires_grad_(False)

        self.updated_model = utils.clone_model(model, requires_grad=True)

        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.n_groups = n_groups

        self.group_size = group_size
        self.learning_rate = learning_rate

        self.reset()

    def reset(self):
        self.current_episode = []

        self.current_log_ratios = []
        self.current_rewards = []

        self.optimizer = optim.Adam(
            params=self.updated_model.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0

    def compute_log_probs(self, sequence, tokenids, model):
        unn_log_probs = model(sequence)
        log_probs = torch.log_softmax(unn_log_probs, dim=-1)

        player_id = (sequence.shape[-1] - 1) % 2
        log_probs = log_probs[:, player_id::2, :]

        return log_probs.gather(-1, tokenids).squeeze(-1).squeeze(0)

    def optimizer_step(self):
        log_ratios = pad_sequence(
            self.current_log_ratios, batch_first=True, padding_value=0.0
        )  # (G, max_len)

        masking_tensor = [
            torch.tensor([True] * ep.shape[0], device=self.engine.device)
            for ep in self.current_log_ratios
        ]
        masks = pad_sequence(masking_tensor, batch_first=True, padding_value=False)
        masks = masks.bool()

        rewards = torch.cat(self.current_rewards).unsqueeze(1)  # (G, 1)
        if rewards.std() < 1e-8:
            print("Group rewards have zero std, skipping update")
            return 0, 0
        elif rewards.std() < 1e-3:
            advantages = rewards - rewards.mean()
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std())

        score1 = torch.exp(log_ratios) * advantages  # (G, max_len)
        score2 = (
            torch.clamp(
                torch.exp(log_ratios), 1 - self.epsilon_low, 1 + self.epsilon_high
            )
            * advantages
        )  # (G, max_len)

        # Masking flattens
        unregularized_score = torch.min(score1, score2)[masks].mean()
        KL = (torch.exp(-log_ratios) + log_ratios - 1)[masks].mean()

        score = unregularized_score - self.beta * KL

        self.optimizer.zero_grad()
        neg_score = -score
        neg_score.backward()
        self.optimizer.step()

        return unregularized_score.item(), KL.item()

    def update(self, state, action, reward, done, next_state, writer=None):
        self.current_episode.append(self.last_tokenid)

        if done:
            self.n_eps += 1

            sequence = mechanics.move_stack_to_sequence(
                state, self.engine.encoder, self.engine.device
            )

            current_episode = np.array(self.current_episode)
            tokenids = torch.tensor(
                current_episode, dtype=torch.int64, device=self.engine.device
            ).transpose(0, 1)  # (1, L, 1)

            self.updated_model.train()
            log_probs = self.compute_log_probs(sequence, tokenids, self.updated_model)
            with torch.no_grad():
                log_probs_old = self.compute_log_probs(
                    sequence, tokenids, self.engine.model
                )

            log_ratio = log_probs - log_probs_old

            self.current_log_ratios.append(log_ratio)
            self.current_rewards.append(
                torch.tensor([reward], dtype=torch.float32, device=self.engine.device)
            )

            self.current_episode = []

            if (self.n_eps % self.group_size) == 0:
                unregularized_score_item, KL_item = self.optimizer_step()
                self.current_log_ratios = []
                self.current_rewards = []

                if writer is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        self.updated_model.parameters(), max_norm=1e9
                    )

                    writer.add_scalar("train/grad_norm", total_norm, self.n_eps)
                    writer.add_scalars(
                        "score",
                        {
                            "unregularized_score": unregularized_score_item,
                            "kl": -self.beta * KL_item,
                        },
                        self.n_eps,
                    )

            if (self.n_eps % (self.group_size * self.n_groups)) == 0:
                self.engine.model.load_state_dict(self.updated_model.state_dict())

        return

    def load_checkpoint(self, checkpoint_name=None):
        rl_checkpoint = super().load_checkpoint(checkpoint_name=checkpoint_name)

        self.optimizer.load_state_dict(rl_checkpoint["optimizer_state_dict"])
        self.lr = rl_checkpoint["lr"]
        self.n_eps = rl_checkpoint["n_eps"]

    def save_checkpoint(self, model_config, checkpoint_name=None):
        rl_checkpoint = {
            "model_config": model_config,
            "encoder": self.engine.encoder,
            "lr": self.learning_rate,
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
