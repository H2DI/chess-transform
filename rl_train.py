from chess_seq.tictactoe.players import (
    RandomPlayer,
    SimplePlayer,
    NNPlayer,
    ReasonablePlayer,
)
from chess_seq.tictactoe.environment import TTTEnv

from chess_seq.tictactoe.GRPO import GRPO
from chess_seq.tictactoe.rl_runner import GRPORunner
from chess_seq.tictactoe.evaluation import full_eval

from torch.utils.tensorboard import SummaryWriter

import chess_seq.utils as utils
from configs import GRPOConfig, ModelConfig

import shutil
import torch
import pickle
import os
import time


if __name__ == "__main__":
    session_config = GRPOConfig()

    writer = SummaryWriter(f"{session_config.log_dir}/{session_config.model_name}")
    device = torch.device(session_config.device_str)
    if session_config.new_model:
        model_config = ModelConfig(name=session_config.model_name)
        model = utils.build_and_save_model(model_config)
        base_name = f"{model_config.name}_GRPO"
    else:
        model, model_config, checkpoint = utils.load_model(session_config.model_name)
        base_name = f"{model_config.name}_{checkpoint['n_games']}"

    with open(model_config.encoder_path, "rb") as f:
        encoder = pickle.load(f)

    logs_path = os.path.join(session_config.log_dir, session_config.model_name)
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path, exist_ok=True)

    agent = GRPO(
        model_config,
        model,
        grpo_config=session_config,
        writer=writer,
    )

    # adversary = SimplePlayer()
    adversary = RandomPlayer()
    # adversary = ReasonablePlayer()

    # adv_model, adv_encoder, _ = utils.load_model(
    #     "ttt_large_573440_GRPO",
    #     # number="50000",
    #     special_name="no_loss",
    # )
    # adversary = NNPlayer(adv_model, adv_encoder, mask_illegal=True, device=None)

    env = TTTEnv(adversary, agent_start=session_config.agent_start)

    full_eval(agent, env, N_eval=250, prints=True, p_start=session_config.p_start)

    runner = GRPORunner(agent, env)

    start_time = time.time()
    runner.train(session_config.max_episodes)
