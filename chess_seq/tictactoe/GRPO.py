import os

import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR


import chess_seq.tictactoe.mechanics as mechanics
from chess_seq.tictactoe.agent import TTTAgent
from torch.utils.tensorboard import SummaryWriter
import chess_seq.utils as utils


class GRPO(TTTAgent):
    """
    Implement the formula for the score:
    \[
        \sum_i \sum_t \min \left(
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

    """

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
        min_lr,
        end_lr_steps,
        device=None,
    ):
        self.full_name = base_name + "_GRPO"
        super().__init__(model, encoder, full_name=self.full_name, device=device)
        self.engine.model.eval()
        for p in self.engine.model.parameters():
            p.requires_grad_(False)

        self.updated_model = utils.clone_model(model, requires_grad=True)

        self.updated_model.to(device)
        self.engine.model.to(device)

        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.n_groups = n_groups

        self.group_size = group_size
        self.learning_rate = learning_rate

        self.total_steps = end_lr_steps
        self.min_lr = min_lr

        self.reset()

    def reset(self):
        self.current_episode = []

        self.group_agentids = []
        self.group_sequences = []
        self.group_tokenids = []
        self.group_rewards = []
        self.group_entropies = []

        self.optimizer = optim.Adam(
            params=self.updated_model.parameters(), lr=self.learning_rate
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(
                self.min_lr / self.learning_rate, 1 - step / self.total_steps
            ),
        )

        self.n_eps = 0

    def group_log_probs(self, sequence, tokenids, agent_ids, model):
        """
        Example:
        agent_id = "X"
        sequence : [ST,  X1,  O1,  X2,  O2 ]
        logits :   [px1, po1, px2, po2, px3]
        tokenids : [X1, X2, X3] taken at indices 0, 2, 4

        agent_id = "O"
        sequence : [ST,  X1,  O1,  X2,  O2,  X3 ]
        logits :   [px1, po1, px2, po2, px3, po3]
        tokenids : [O1, O2, O3] taken at indices 1, 3, 5

        [ST, X11, O11, X12, O12, X13]
        [ST, X21, O21, X22, O22, X23]
        [ST, X31, O31, X32, pad, pad]
        [ST, X41, O41, X42, O42, X43]
        agent_ids = ["X", "O", "X", "O"]
        tokenids                   indices
        [[X11, X12, X13],          [1, 3, 5],
         [O21, O22, pad],          [2, 4, pad],
         [X31, X32, pad],          [1, 3, 5],
         [O41, O42, pad]]          [2, 4, pad]]

        Since tokenids ids are padded, we can put a dummy value when gathering the logits.
        The later masking will ignore these values.

        TODO: TEST THIS CAREFULLY. With even and odd sequence lengths.
        """
        unn_log_probs = model(sequence)  # B, T, vocab_size
        log_probs = torch.log_softmax(unn_log_probs, dim=-1)

        L = log_probs.shape[1] // 2 + 1
        start_indices = np.array([0 if i == "X" else 1 for i in agent_ids])  # (B,)
        idx = (
            start_indices[:, None, None] + np.arange(L)[None, :, None] * 2
        )  # (B, L, 1)
        idx = idx + np.arange(log_probs.shape[2])[None, None, :]  # (B, L, vocab_size)
        idx = torch.tensor(idx, device=log_probs.device)
        idx = idx.clamp(0, log_probs.shape[1] - 1)
        log_probs = log_probs.gather(1, idx)  # (B, L, vocab_size)
        return log_probs.gather(-1, tokenids.unsqueeze(-1)).squeeze(-1)  # (B, L)

    def optimizer_step(self, writer: SummaryWriter = None):
        sequences_tensor = pad_sequence(
            self.group_sequences,
            batch_first=True,
            padding_value=self.engine.end_tokenid,
        )  # (G, max_len)

        group_tokenids = pad_sequence(
            self.group_tokenids, batch_first=True, padding_value=0
        )

        log_probs = self.group_log_probs(
            sequences_tensor, group_tokenids, self.group_agentids, self.updated_model
        )  # (G, max_len)

        with torch.no_grad():
            old_log_probs = self.group_log_probs(
                sequences_tensor, group_tokenids, self.group_agentids, self.engine.model
            )
        log_ratios = log_probs - old_log_probs  # (G, max_len)

        lengths = [ep.shape[0] for ep in self.group_tokenids]
        max_len = max(lengths)
        masks = torch.zeros(
            (len(lengths), max_len), dtype=torch.bool, device=self.engine.device
        )
        for i, length in enumerate(lengths):
            masks[i, :length] = True

        rewards = torch.cat(self.group_rewards).unsqueeze(1)  # (G, 1)
        if rewards.std() < 1e-8:
            print("Group rewards have zero std, skipping update")
            return 0, 0
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-3)

        score1 = torch.exp(log_ratios) * advantages  # (G, max_len)
        score2 = (
            torch.clamp(
                torch.exp(log_ratios), 1 - self.epsilon_low, 1 + self.epsilon_high
            )
            * advantages
        )  # (G, max_len)

        # Masking flattens
        unregularized_score = torch.minimum(score1, score2)[masks].mean()
        KL = (torch.exp(-log_ratios) + log_ratios - 1)[masks].mean()

        score = unregularized_score - self.beta * KL

        self.optimizer.zero_grad()
        neg_score = -score
        neg_score.backward()
        torch.nn.utils.clip_grad_norm_(self.updated_model.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        if writer is not None:
            self.log_training(writer, KL.item(), unregularized_score.item())
            lr = self.scheduler.get_last_lr()[0]
            writer.add_scalar("train/learning_rate", lr, self.n_eps)

        return

    def log_training(self, writer, KL_item, unregularized_score_item):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.updated_model.parameters(), max_norm=1e9
        )

        weights_norm = torch.norm(
            torch.stack([p.norm(2) for p in self.updated_model.parameters()])
        ).item()
        writer.add_scalar("train/weights_norm", weights_norm, self.n_eps)
        writer.add_scalar("train/grad_norm", grad_norm, self.n_eps)
        writer.add_scalars(
            "score",
            {
                "unregularized_score": unregularized_score_item,
                "kl": -self.beta * KL_item,
            },
            self.n_eps,
        )

    def update(self, state, action, reward, done, next_state, writer=None):
        self.current_episode.append(self.last_tokenid[0, 0])
        self.group_entropies.append(self.last_token_entropy)

        if done:
            self.n_eps += 1

            self.group_agentids.append(self.agent_id)

            self.group_rewards.append(
                torch.tensor([reward], dtype=torch.float32, device=self.engine.device)
            )

            sequence = mechanics.move_stack_to_sequence(
                state, self.engine.encoder, self.engine.device
            )[0]  # (1, L)
            self.group_sequences.append(sequence)

            self.group_tokenids.append(
                torch.tensor(
                    self.current_episode, dtype=torch.int64, device=self.engine.device
                )
            )

            self.current_episode = []

            if (self.n_eps % self.group_size) == 0:
                self.optimizer_step(writer=writer)
                if writer is not None:
                    writer.add_scalar(
                        "train/entropies", np.mean(self.group_entropies), self.n_eps
                    )

                self.group_agentids = []
                self.group_tokenids = []
                self.group_sequences = []
                self.group_rewards = []
                self.group_entropies = []

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
