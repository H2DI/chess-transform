import os

import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter

import chess_seq.tictactoe.mechanics as mechanics
from chess_seq.tictactoe.agent import TTTAgent
from chess_seq.utils import clone_model
from configs import GRPOConfig, ModelConfig


class GRPO(TTTAgent):
    """
    Implement the formula for the score:
    \[
        \sum \min \left(
            \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t,
                \text{clip}\left(
                        \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)},
                        1 - \epsilon,
                        1 + \epsilon
                    \right)
                    A_t
        \right)
        - \beta KL\left(\pi_\theta\right) ||\pi_{\theta_{old}})
    \]

    Using practice recommendations from:
    https://arxiv.org/abs/2503.20783
    """

    def __init__(
        self,
        model_config: ModelConfig,
        model: torch.nn.Module,
        grpo_config: GRPOConfig,
        writer: SummaryWriter = None,
    ):
        self.full_name = grpo_config.model_name + "_GRPO"
        device = torch.device(grpo_config.device_str)
        super().__init__(model_config, model, full_name=self.full_name, device=device)
        self.engine.model.to(device)
        self.engine.model.eval()
        for p in self.engine.model.parameters():
            p.requires_grad_(False)

        self.updated_model = clone_model(model, requires_grad=True)
        self.updated_model.to(device)
        self.updated_model.eval()

        self.rollout_temperature = grpo_config.rollout_temperature

        self.beta = grpo_config.beta
        self.epsilon_low = grpo_config.epsilon_low
        self.epsilon_high = grpo_config.epsilon_high
        self.temperature = grpo_config.rollout_temperature

        self.group_size = grpo_config.group_size
        self.groups_between_prompts = grpo_config.groups_between_prompts
        self.prompts_between_models = grpo_config.prompts_between_models

        self.learning_rate = grpo_config.learning_rate
        self.min_lr = grpo_config.min_lr
        self.total_steps = grpo_config.end_lr_steps

        self.eval_frequency = grpo_config.eval_frequency

        self.device = device
        self.writer = writer
        self.prints = grpo_config.debug_prints
        self.p_start = grpo_config.p_start

        self.reset()

    def reset(self):
        self.n_eps = 0
        self.current_episode = []
        self._clean_group_data()

        self.optimizer = optim.Adam(
            params=self.updated_model.parameters(), lr=self.learning_rate
        )
        # self.scheduler = LambdaLR(
        #     self.optimizer,
        #     lr_lambda=lambda step: max(
        #         self.min_lr / self.learning_rate, 1 - step * self.group_size / self.total_steps
        #     ),
        # )
        self.lr_lambda = lambda step: self.min_lr / self.learning_rate + 0.5 * (
            1 - self.min_lr / self.learning_rate
        ) * (1 + np.cos(np.pi * min(1, step * self.group_size / self.total_steps)))
        # * (
        #     1 / 3 + (2 / 3) * max(0, 1 - step * self.group_size / (3 * self.total_steps))
        # )
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

    def update(self):
        self.current_episode.append(self.last_tokenid[0, 0])
        self.group_entropies.append(self.last_token_entropy)

    def end_episode_update(self, state, reward):
        self.n_eps += 1
        rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.group_rewards.append(rewards)

        sequence = mechanics.move_stack_to_sequence(
            state, self.engine.encoder, self.device
        )[0]
        self.group_sequences.append(sequence)

        self.group_tokenids.append(
            torch.tensor(self.current_episode, dtype=torch.int64, device=self.device)
        )

        self.current_episode = []

    def end_group_update(self):
        self._optimizer_step()
        self._log_entropy()
        self._clean_group_data()

    def copy_updated_to_actor(self):
        if self.prints:
            print("Copying updated model to actor model")
        self.engine.model.load_state_dict(self.updated_model.state_dict())

    def _optimizer_step(self):
        """
        9 is the padding token. Mask removes the padding tokens in the score computation. Example with group_size=3. We want to compute
        pi(a_t|s_t) for t the time steps where the agent played.
        """
        log_ratios = self.compute_log_ratios()  # (G, L)
        masks = self.compute_mask()  # (G, L)
        rewards = torch.cat(self.group_rewards).unsqueeze(1)  # (G, 1)

        if rewards.std() < 1e-8:
            print("Group rewards have zero std, skipping update")
            return
        else:
            advantages = rewards - rewards.mean()  # / (rewards.std() + 1e-3)

        if self.prints:
            print(f"log_ratios (should be ~0 if models are identical):\n{log_ratios}")
            print(f"{rewards=}")
            print(f"{masks=}")
            print("----------------------")

        ratios = torch.exp(log_ratios)  # (G, L)
        score1 = ratios * advantages
        score2 = ratios.clamp(1 - self.epsilon_low, 1 + self.epsilon_high) * advantages

        # Masking flattens / modern version: all tokens count equally
        unregularized_score = torch.minimum(score1, score2)[masks].mean()
        KL = (torch.exp(-log_ratios) + log_ratios - 1)[masks].mean()

        score = unregularized_score - self.beta * KL

        self.optimizer.zero_grad()
        neg_score = -score

        neg_score.backward()
        torch.nn.utils.clip_grad_norm_(self.updated_model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        self._log_training(
            KL.item(),
            unregularized_score.item(),
            rewards.cpu().numpy().mean(),
            rewards.cpu().numpy().std(),
        )

    def compute_log_ratios(self):
        group_sequence = pad_sequence(
            self.group_sequences,
            batch_first=True,
            padding_value=self.engine.end_tokenid,
        )  # (G, max_len), L = max_len // 2 + max_len % 2

        group_tokenids = pad_sequence(
            self.group_tokenids, batch_first=True, padding_value=self.engine.end_tokenid
        )  # (G, L)
        if self.prints:
            print(f"{group_sequence=}")
            print(f"{group_tokenids=}")

        log_probs = self.group_log_probs(
            group_sequence,
            group_tokenids,
            self.updated_model,
            prints=self.prints,
        )  # (G, L)

        with torch.no_grad():
            old_log_probs = self.group_log_probs(
                group_sequence, group_tokenids, self.engine.model
            )
        return log_probs - old_log_probs  # (G, L)

    def group_log_probs(
        self,
        sequence: torch.Tensor,
        tokenids: torch.Tensor,
        model: torch.nn.Module,
        prints=False,
    ):
        """ """
        T = sequence.shape[1]
        causal_mask = torch.tril(torch.ones(T, T), diagonal=0).to(self.device).bool()
        unn_log_probs = model(sequence, mask=causal_mask)  # G, T, V
        log_probs = torch.log_softmax(unn_log_probs, dim=-1)

        played_log_probs = self.grab_play_indices(log_probs, prints)  # (G, L, V)

        return played_log_probs.gather(-1, tokenids.unsqueeze(-1)).squeeze(-1)  # (G, L)

    def grab_play_indices(self, log_probs: torch.Tensor, prints=False):
        """
        Given tensor of log_probs of shape (B, T, vocab_size)
        and start_index with value 0 or 1
        returns the log_probs at indices [i, 2*j + start_index, :]
        which does not matter since these will be masked out later)
        """
        start_index = 0 if self.agent_id == "X" else 1
        max_len = log_probs.shape[1]

        # remove 1 if max_len even and agent is O (len 10 -> 4 moves for O)
        L = max_len // 2 - (1 - max_len % 2) * start_index
        idx = torch.arange(L, device=self.device).unsqueeze(0) * 2 + start_index

        idx = idx.unsqueeze(-1).expand(
            log_probs.shape[0], -1, log_probs.shape[2]
        )  # (G, L, vocab_size)
        if prints:
            print(f"{self.agent_id=}")
            print(f"{log_probs.shape=}")
            print(f"{idx=}")
            print(f"{idx.shape=}")
            print(f"{log_probs.gather(1, idx).shape=}")
        return log_probs.gather(1, idx)  # (G, L, vocab_size)

    def compute_mask(self):
        lengths = [ep.shape[0] for ep in self.group_tokenids]
        max_len = max(lengths)
        masks = torch.zeros(
            (len(lengths), max_len), dtype=torch.bool, device=self.device
        )
        for i, length in enumerate(lengths):
            masks[i, :length] = True
        return masks

    def _clean_group_data(self):
        self.group_tokenids = []
        self.group_sequences = []
        self.group_rewards = []
        self.group_entropies = []

    def _log_training(self, KL_item, unregularized_score_item, reward_mean, reward_std):
        if self.writer is not None:
            self.writer.add_scalar("score/group_rewards_mean", reward_mean, self.n_eps)
            self.writer.add_scalar("score/group_rewards_std", reward_std, self.n_eps)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.updated_model.parameters(), max_norm=1e9
            )

            weights_norm = torch.norm(
                torch.stack([p.norm(2) for p in self.updated_model.parameters()])
            ).item()
            self.writer.add_scalar("train/weights_norm", weights_norm, self.n_eps)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.n_eps)
            self.writer.add_scalars(
                "score",
                {
                    "unregularized_score": unregularized_score_item,
                    "kl": -self.beta * KL_item,
                },
                self.n_eps,
            )

            lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar("train/learning_rate", lr, self.n_eps)

    def _log_entropy(self):
        if self.writer is not None:
            self.writer.add_scalar(
                "train/entropies", np.mean(self.group_entropies), self.n_eps
            )

    def load_checkpoint(self, checkpoint_name=None):
        rl_checkpoint = super().load_checkpoint(checkpoint_name=checkpoint_name)

        self.optimizer.load_state_dict(rl_checkpoint["optimizer_state_dict"])
        self.lr = rl_checkpoint["lr"]
        self.n_eps = rl_checkpoint["n_eps"]

    def save_checkpoint(self, checkpoint_name=None):
        rl_checkpoint = {
            "model_config": self.engine.model_config,
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
