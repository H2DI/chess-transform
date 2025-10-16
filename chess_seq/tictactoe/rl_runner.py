from chess_seq.tictactoe.GRPO import GRPO
from chess_seq.tictactoe.environment import TTTEnv


from chess_seq.tictactoe.evaluation import full_eval
import time


class GRPORunner:
    def __init__(self, agent: GRPO, env: TTTEnv):
        self.agent = agent
        self.env = env

        self.group_size = self.agent.group_size
        self.groups_between_prompts = self.agent.groups_between_prompts
        self.prompts_between_models = self.agent.prompts_between_models
        self.p_start = self.agent.p_start

        self.eval_frequency = max(
            self.group_size * self.groups_between_prompts, self.agent.eval_frequency
        )
        self.ep_i = 0

    def train(self, max_episodes):
        self.start_time = time.time()
        while self.ep_i < max_episodes:
            for _ in range(self.prompts_between_models):
                self.env.set_new_prompt()
                for _ in range(self.groups_between_prompts):
                    self.rollout_group()
                    self.agent.end_group_update()
            self.agent.copy_updated_to_actor()

            if (self.ep_i + 1) % self.eval_frequency == 1 and self.ep_i > 0:
                keep_going = self.short_eval()
                if not keep_going:
                    break

            if (self.ep_i + 1) % (5 * self.eval_frequency) == 0:
                full_eval(
                    self.agent, self.env, N_eval=250, prints=True, p_start=self.p_start
                )
                self.agent.save_checkpoint()

    def rollout_group(self):
        for _ in range(self.group_size):
            self.run_episode()
            self.ep_i += 1

    def run_episode(self):
        state, info = self.env.reset_to_prompt()
        self.agent.new_game(info["agent_id"])
        legal_moves = info.get("legal_moves", [])
        done = False
        while not done:
            action = self.agent.get_action(state, legals=legal_moves)
            state, reward, terminated, truncated, info = self.env.step(action)
            legal_moves = info.get("legal_moves", [])
            done = terminated or truncated
            self.agent.update()
        self.agent.end_episode_update(state, reward)

    def short_eval(self):
        print(f"Train time: {time.time() - self.start_time:.1f} seconds")
        wins, losses, _, illegal_moves = full_eval(
            self.agent, self.env, N_eval=250, p_start=self.p_start
        )
        if illegal_moves > 0.1:
            print("Too many illegal moves")
            return False
        if losses < 0.05 and wins > 0.8 and illegal_moves == 0:
            print("Starting Sub10 evaluation on 500 games")
            wins, losses, ties, illegal_moves = full_eval(
                self.agent,
                self.env,
                N_eval=250,
                prints=True,
                p_start=self.p_start,
            )
            if wins == 1:
                self.agent.save_checkpoint(checkpoint_name="beat")
                print("Adversary solved")
                return False
        return True
