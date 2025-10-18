import numpy as np

from chess_seq.tictactoe.environment import TTTEnv
from chess_seq.tictactoe.agent import TTTAgent
from torch import no_grad


@no_grad()
def play_one_game(agent: TTTAgent, env: TTTEnv, agent_start=True, temperature=0.0):
    state, _ = env.reset(agent_start=agent_start)
    agent_id = "X" if agent_start else "O"
    agent.new_game(agent_id)
    legal_moves = env.game.legal_moves

    done = False
    while not done:
        action = agent.get_action(state, temperature=temperature, legals=legal_moves)
        state, reward, terminated, truncated, info = env.step(action)
        legal_moves = info.get("legal_moves", [])
        done = terminated or truncated
    return reward, truncated, env.game.winner


@no_grad()
def evaluate_agent(
    agent: TTTAgent,
    env: TTTEnv,
    N_eval=100,
    prints=False,
    agent_start=True,
    log_illegal=False,
):
    if N_eval == 0:
        return 0, 0, 0, 0
    agent.engine.model.eval()
    rewards = []
    scores = []
    illegal_moves = []
    agent_id = "X" if agent_start else "O"
    for _ in range(N_eval):
        reward, truncated, result = play_one_game(
            agent, env, agent_start=agent_start, temperature=0.0
        )
        rewards.append(reward)
        if result == "T":
            scores.append(0)
        elif result == agent_id:
            scores.append(1)
        else:
            scores.append(-1)
        illegal_moves.append(truncated)
    scores = np.array(scores)
    wins = np.sum(scores == 1) / N_eval
    losses = np.sum(scores == -1) / N_eval
    ties = np.sum(scores == 0) / N_eval
    illegal_moves = np.sum(illegal_moves) / N_eval
    agent.writer.add_scalar(f"eval{agent_id}/wins", wins, agent.n_eps)
    agent.writer.add_scalar(f"eval{agent_id}/losses", losses, agent.n_eps)
    agent.writer.add_scalar(f"eval{agent_id}/ties", ties, agent.n_eps)
    if log_illegal:
        agent.writer.add_scalar(
            f"eval{agent_id}/illegal_moves", illegal_moves, agent.n_eps
        )
    agent.writer.add_scalar(f"eval{agent_id}/reward", np.mean(rewards), agent.n_eps)
    if prints:
        print(f"Evaluation results over {N_eval} games, player {agent_id}:")
        print(f"Wins: {wins * 100:.2f}%")
        print(f"Losses: {losses * 100:.2f}%")
        print(f"Ties: {ties * 100:.2f}%")
        if log_illegal:
            print(f"Illegal moves: {illegal_moves * 100:.2f}%")
    return wins, losses, ties, illegal_moves


def full_eval(
    agent, env: TTTEnv, N_eval=200, prints=False, p_start=0.5, log_illegal=False
):
    if prints:
        print(f"Starting full evaluation after {agent.n_eps} games played.")

    if p_start > 0:
        print("New game as X")
        play_one_game(agent, env, agent_start=True, temperature=0.0)
        env.game.print_game()
    if p_start < 1:
        print("New game as O")
        play_one_game(agent, env, agent_start=False, temperature=0.0)
        env.game.print_game()

    xwins, xlosses, xties, xills = evaluate_agent(
        agent,
        env,
        N_eval=int(N_eval * p_start),
        prints=prints,
        agent_start=True,
        log_illegal=log_illegal,
    )
    owins, olosses, oties, oills = evaluate_agent(
        agent,
        env,
        N_eval=int((1 - p_start) * N_eval),
        prints=prints,
        agent_start=False,
        log_illegal=log_illegal,
    )
    p = p_start
    return (
        p * xwins + (1 - p) * owins,
        p * xlosses + (1 - p) * olosses,
        p * xties + (1 - p) * oties,
        p * xills + (1 - p) * oills,
    )
