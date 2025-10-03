from chess_seq.tictactoe.players import RandomPlayer, SimplePlayer, NNPlayer
from chess_seq.tictactoe.environment import TTTEnv

# from chess_seq.tictactoe.reinforce import REINFORCE
from chess_seq.tictactoe.GRPO import GRPO

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import chess_seq.utils as utils
from configs import RLTraining
from torch import no_grad
import shutil
import os


@no_grad()
def evaluate_agent(agent, env, writer, N_eval=100, prints=False, agent_start=True):
    agent.engine.model.eval()
    results = []
    illegal_moves = []
    player_id = "X" if agent_start else "O"
    for _ in range(N_eval):
        state, _ = env.reset(agent_start=agent_start)
        agent.new_game()
        done = False
        while not done:
            action = agent.get_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                results.append(reward)
                illegal_moves.append(truncated)
    results = np.array(results)
    wins = np.sum(results == 1) / N_eval
    losses = np.sum(results == -1) / N_eval
    ties = np.sum(results == 0) / N_eval
    illegal_moves = np.sum(illegal_moves) / N_eval
    writer.add_scalar(f"eval{player_id}/wins", wins, agent.n_eps)
    writer.add_scalar(f"eval{player_id}/losses", losses, agent.n_eps)
    writer.add_scalar(f"eval{player_id}/ties", ties, agent.n_eps)
    writer.add_scalar(f"eval{player_id}/illegal_moves", illegal_moves, agent.n_eps)
    writer.add_scalar(f"eval{player_id}/reward", np.mean(results), agent.n_eps)
    if prints:
        print(f"Evaluation results over {N_eval} games, player {player_id}:")
        print(f"Wins: {wins * 100:.2f}%")
        print(f"Losses: {losses * 100:.2f}%")
        print(f"Ties: {ties * 100:.2f}%")
        print(f"Illegal moves: {illegal_moves * 100:.2f}%")
    agent.engine.model.train()
    return wins, losses, ties, illegal_moves


def full_eval(agent, env, writer, N_eval=200, prints=False):
    if prints:
        print("Starting full evaluation...")
    xwins, xlosses, xties, xills = evaluate_agent(
        agent, env, writer, N_eval=N_eval, prints=prints, agent_start=True
    )
    owins, olosses, oties, oills = evaluate_agent(
        agent, env, writer, N_eval=N_eval, prints=prints, agent_start=False
    )
    return (
        (xwins + owins) / 2,
        (xlosses + olosses) / 2,
        (xties + oties) / 2,
        (xills + oills) / 2,
    )


if __name__ == "__main__":
    session = RLTraining()
    writer = SummaryWriter(f"{session.log_dir}/{session.model_name}")

    model, encoder, checkpoint = utils.load_model(session.model_name)
    model_config = checkpoint["model_config"]
    base_name = f"{model_config.name}_{checkpoint['n_games']}"

    logs_path = os.path.join(session.log_dir, session.model_name)
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path, exist_ok=True)

    agent = GRPO(
        model,
        encoder,
        base_name=base_name,
        beta=0.2,
        epsilon_low=0.1,
        epsilon_high=0.1,
        group_size=16,
        n_groups=4,
        learning_rate=5e-6,
    )

    # adversary = SimplePlayer()
    adversary = RandomPlayer()

    # adv_model, adv_encoder, _ = utils.load_model(
    #     "ttt_large_573440_GRPO",
    #     number="50000",
    #     # special_name="sub10",
    # )
    # adversary = NNPlayer(adv_model, adv_encoder, mask_illegal=True, device=None)

    env = TTTEnv(adversary, agent_start=None, greedy_adversary=True, illegal_cost=-5)
    max_steps = 400_000

    print("Initial performance: ")
    full_eval(agent, env, writer, N_eval=250, prints=True)

    for i in range(max_steps):
        state, _ = env.reset()
        agent.new_game()
        done = False
        while not done:
            action = agent.get_action(state, greedy=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, done, next_state, writer=writer)
            state = next_state

        if (i + 1) % 1000 == 1 and i > 0:
            _, losses, _, illegal_moves = full_eval(agent, env, writer, N_eval=250)
            if illegal_moves > 0.1:
                print("Too many illegal moves")
                break
            if losses < 0.10 and illegal_moves == 0:
                print("Starting Sub10 evaluation on 500 games")
                wins, losses, ties, illegal_moves = full_eval(
                    agent, env, writer, N_eval=250, prints=True
                )
                if losses < 0.10 and illegal_moves == 0:
                    agent.save_checkpoint(model_config, checkpoint_name="sub10")
                    print("Sub10 model saved")
                    break

        if (i + 1) % 5000 == 0:
            full_eval(agent, env, writer, N_eval=250, prints=True)
            agent.save_checkpoint(model_config)
