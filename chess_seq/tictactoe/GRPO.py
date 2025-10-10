import os

import numpy as np
import torch
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import chess_seq.tictactoe.mechanics as mechanics
from chess_seq.tictactoe.agent import TTTAgent
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

        self.current_log_ratios = []
        self.current_rewards = []
        self.current_entropies = []

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

    def compute_log_probs(self, sequence, tokenids, model):
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

        """
        unn_log_probs = model(sequence)
        log_probs = torch.log_softmax(unn_log_probs, dim=-1)

        start_index = 0 if self.agent_id == "X" else 1
        log_probs = log_probs[:, start_index::2, :]

        return log_probs.gather(-1, tokenids).squeeze(-1).squeeze(0)

    def optimizer_step(self, writer=None):
        log_ratios = pad_sequence(
            self.current_log_ratios, batch_first=True, padding_value=0.0
        )  # (G, max_len)

        lengths = [ep.shape[0] for ep in self.current_log_ratios]
        max_len = max(lengths)
        masks = torch.zeros(
            (len(lengths), max_len), dtype=torch.bool, device=self.engine.device
        )
        for i, length in enumerate(lengths):
            masks[i, :length] = True

        rewards = torch.cat(self.current_rewards).unsqueeze(1)  # (G, 1)
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
        self.current_episode.append(self.last_tokenid)
        self.current_entropies.append(self.last_token_entropy)

        if done:
            self.n_eps += 1

            self.current_rewards.append(
                torch.tensor([reward], dtype=torch.float32, device=self.engine.device)
            )

            tokenids = torch.tensor(
                np.array(self.current_episode),
                dtype=torch.int64,
                device=self.engine.device,
            ).transpose(0, 1)  # (1, L, 1)

            self.updated_model.train()
            sequence = mechanics.move_stack_to_sequence(
                state, self.engine.encoder, self.engine.device
            )
            log_probs = self.compute_log_probs(sequence, tokenids, self.updated_model)
            with torch.no_grad():
                log_probs_old = self.compute_log_probs(
                    sequence, tokenids, self.engine.model
                )

            log_ratio = log_probs - log_probs_old
            self.current_log_ratios.append(log_ratio)

            self.current_episode = []

            if (self.n_eps % self.group_size) == 0:
                self.optimizer_step(writer=writer)
                if writer is not None:
                    writer.add_scalar(
                        "train/entropies", np.mean(self.current_entropies), self.n_eps
                    )
                self.current_log_ratios = []
                self.current_rewards = []
                self.current_entropies = []

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
