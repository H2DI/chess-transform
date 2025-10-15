from chess_seq.tictactoe.players import (
    RandomPlayer,
    SimplePlayer,
    NNPlayer,
    ReasonablePlayer,
)
from chess_seq.tictactoe.environment import TTTEnv

# from chess_seq.tictactoe.reinforce import REINFORCE
from chess_seq.tictactoe.GRPO import GRPO
from chess_seq.tictactoe.evaluation import full_eval

from torch.utils.tensorboard import SummaryWriter

import chess_seq.utils as utils
import chess_seq.models as models
from configs import RLTraining, ModelConfig

import shutil
import torch
import pickle
import os
import time


if __name__ == "__main__":
    session = RLTraining()

    writer = SummaryWriter(f"{session.log_dir}/{session.model_name}")
    device = torch.device(session.device_str)

    model, encoder, checkpoint = utils.load_model(session.model_name)
    model_config = checkpoint["model_config"]
    base_name = f"{model_config.name}_{checkpoint['n_games']}"
    model = models.ChessNet(config=model_config).to(device)

    # model_config = ModelConfig()
    # model = models.ChessNet(config=model_config).to(device)

    with open(model_config.encoder_path, "rb") as f:
        encoder = pickle.load(f)
    base_name = f"{model_config.name}_GRPO"

    logs_path = os.path.join(session.log_dir, session.model_name)
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path, exist_ok=True)

    group_size = 4
    n_groups = 64
    end_lr_steps = 40000 / (group_size)

    eval_frequency = max(1000, group_size * n_groups)
    rollout_temp = 0.5

    agent = GRPO(
        model,
        encoder,
        device=device,
        base_name=base_name,
        beta=0,
        epsilon_low=0.3,
        epsilon_high=0.3,
        group_size=group_size,
        n_groups=n_groups,
        learning_rate=5e-5,
        min_lr=1e-5,
        end_lr_steps=end_lr_steps,
        writer=writer,
        prints=True,
    )

    # agent = REINFORCE(
    #     model,
    #     encoder,
    #     base_name=base_name,
    #     gamma=1,
    #     episode_batch_size=32,
    #     learning_rate=1e-6,
    # )

    # adversary = SimplePlayer()
    adversary = RandomPlayer()
    # adversary = ReasonablePlayer()

    # adv_model, adv_encoder, _ = utils.load_model(
    #     "ttt_large_573440_GRPO",
    #     # number="50000",
    #     special_name="no_loss",
    # )
    # adversary = NNPlayer(adv_model, adv_encoder, mask_illegal=True, device=None)

    env = TTTEnv(adversary, agent_start=None, illegal_cost=-5)
    max_episodes = 400_000

    p_start = 0.5
    full_eval(agent, env, writer, N_eval=250, prints=True, p_start=p_start)

    start_time = time.time()

    ep_i = 0
    group_i = 0
    while ep_i < max_episodes:
        agent_start = True if (group_i % 2 == 0) else False
        group_i += 1
        for j in range(group_size):
            ep_i += 1
            state, info = env.reset(agent_start=agent_start)
            agent.new_game(info["agent_id"])
            done = False
            legal_moves = env.game.legal_moves
            while not done:
                action = agent.get_action(
                    state, temperature=rollout_temp, legals=legal_moves
                )
                next_state, reward, terminated, truncated, info = env.step(action)
                legal_moves = info.get("legal_moves", [])
                done = terminated or truncated
                agent.update(state, action, reward, done, next_state)
                state = next_state

        if (ep_i + 1) % eval_frequency == 1 and ep_i > 0:
            print(f"Train time: {time.time() - start_time:.1f} seconds")
            wins, losses, _, illegal_moves = full_eval(
                agent, env, writer, N_eval=250, p_start=p_start
            )
            if illegal_moves > 0.1:
                print("Too many illegal moves")
                break
            if losses < 0.05 and wins > 0.8 and illegal_moves == 0:
                print("Starting Sub10 evaluation on 500 games")
                wins, losses, ties, illegal_moves = full_eval(
                    agent, env, writer, N_eval=250, prints=True, p_start=p_start
                )
                if losses == 0 and illegal_moves == 0:
                    agent.save_checkpoint(model_config, checkpoint_name="no_loss")
                    # print("Beat adversary")
                    print("No_loss model saved")
                    break

        if (ep_i + 1) % (5 * eval_frequency) == 0:
            full_eval(agent, env, writer, N_eval=250, prints=True, p_start=p_start)
            agent.save_checkpoint(model_config)
