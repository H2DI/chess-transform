import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.no_grad import no_grad

from chess_seq.tictactoe.agent import TTTAgent
from chess_seq.tictactoe.mechanics import TTTBoard
from chess_seq.tictactoe.reinforce import REINFORCE


class PPO(REINFORCE):
    """
    WIP
    """

    def __init__(
        self,
        actor,
        critic,
        encoder,
        gamma,
        episode_batch_size,
        actor_learning_rate,
        critic_learning_rate,
        lambda_=0.95,
        writer=None,
    ):
        super().__init__(actor, encoder)
        self.critic = critic
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = 0.2

        self.episode_batch_size = episode_batch_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        self.loss_function = nn.MSELoss()
        self.writer = writer

        self.optimizer = optim.Adam(
            params=self.actor.parameters(), lr=self.actor_learning_rate
        )

        self.critic_optimizer = optim.Adam(
            params=self.critic.parameters(), lr=self.critic_learning_rate
        )
        self.current_episode = []
        self.episode_reward = 0

        self.scores = []

        self.n_eps = 0
        self.total_steps = 0
        self.critic_updates = 0

    def compute_GAE(self, rewards, terminateds, advantages):
        """
        Generalized Advantage Estimation
        """
        GAE = 0
        GAE_list = []
        for t in reversed(range(len(rewards))):
            GAE = (1 - terminateds[t]) * GAE
            GAE = advantages[t] + self.gamma * self.lambda_ * GAE
            GAE_list.append(GAE)
        return torch.tensor(GAE_list[::-1], dtype=torch.float32)

    def compute_ppo_score(self, states):
        actions, rewards, terminals, next_states, old_log_probs = tuple(
            [torch.cat(data) for data in zip(*self.current_episode)]
        )

        with torch.no_grad():
            target_values = (
                rewards
                + self.gamma * (1 - terminals) * self.critic(next_states).squeeze()
            )
            values = self.critic(states).squeeze()
            advantages = target_values - values

        GAEs = self.compute_GAE(rewards, terminals, advantages)
        GAEs = (GAEs - GAEs.mean()) / GAEs.std()

        logits = self.actor(states)
        log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        log_probs = log_probs.gather(1, actions).squeeze()
        ratio = torch.exp(log_probs - old_log_probs)

        clipped_ratio = torch.clamp(ratio, min=1 - self.eps, max=1 + self.eps)
        ppo_clip_obj = torch.min(ratio * GAEs, clipped_ratio * GAEs)

        # if self.writer:
        #     probs = torch.softmax(logits, dim=-1)
        #     entropy = -(probs * log_probs).sum(dim=1).mean()
        #     self.writer.add_scalar("policy/entropy", entropy.item(), self.n_eps)

        return ppo_clip_obj.sum().unsqueeze(0)

    def train_reset(self):
        self.current_episode = []
        self.episode_reward = 0
        self.scores = []

    def update_critic(self, transition):
        state, _, reward, terminated, next_state, _ = transition

        values = self.critic.forward(state)
        with torch.no_grad():
            next_state_values = (1 - terminated) * self.critic(next_state)
            targets = next_state_values * self.gamma + reward

        loss = self.loss_function(values, targets.unsqueeze(1))
        if self.writer:
            self.writer.add_scalar("loss/critic", loss.item(), self.total_steps)

        self.critic_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(self.critic.parameters(), 5)
        self.critic_optimizer.step()

    def update(self, state, action, reward, terminated, next_state):
        """
        **
         Formula:
        \sum_{t=0}^{T} \gamma^t  \nabla_\theta \log \pi_\theta(a_t | s_t)
        (R_{t + 1} + \gamma V(S_{t+1}) - V(s_t))
         **
        """
        with torch.no_grad():
            old_logits = self.actor(torch.tensor(state).unsqueeze(0))
            old_log_probs = old_logits - torch.logsumexp(
                old_logits, dim=1, keepdim=True
            )
            old_log_probs = old_log_probs[0, action].unsqueeze(0)

        transition = (
            torch.tensor(state).unsqueeze(0),
            torch.tensor(action, dtype=torch.int64).unsqueeze(0).unsqueeze(0),
            torch.tensor([reward]),
            torch.tensor([terminated], dtype=torch.int64),
            torch.tensor(next_state).unsqueeze(0),
            old_log_probs,
        )

        self.total_steps += 1
        self.episode_reward += reward

        self.current_episode.append(transition)
        self.update_critic(transition)

        if terminated:
            self.writer.add_scalar("policy/reward", self.episode_reward, self.n_eps)
            self.episode_reward = 0
            self.n_eps += 1

            self.scores.append(self.compute_ppo_score())
            self.current_episode = []

            if (self.n_eps % self.episode_batch_size) == 0:
                self.optimizer.zero_grad()
                full_neg_score = -torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()
                self.optimizer.step()
                if self.writer:
                    self.writer.add_scalar(
                        "loss/actor", full_neg_score.item(), self.n_eps
                    )

                self.scores = []
