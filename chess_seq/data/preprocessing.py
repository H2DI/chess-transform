import chess.pgn
import os

from chess_seq.encoder import MoveEncoder
import numpy as np


def save_shard(shard_path, game_ids, tokens):
    """Write a shard to disk."""
    np.savez_compressed(
        shard_path,
        game_ids=np.array(game_ids, dtype=np.int32),
        tokens=np.array(tokens, dtype=object),
    )
    print(f"[saved] {shard_path} with {len(game_ids)} games")


def process_pgn_file(
    input_file, shard_dir, encoder: MoveEncoder, shard_start_index=0, global_game_id=0
):
    """
    Process a PGN file and write to multiple shards in shard_dir.
    Returns:
        next_shard_index, next_global_game_id
    """
    shard_game_ids = []
    shard_tokens = []
    shard_idx = shard_start_index

    def shard_path(idx):
        return os.path.join(shard_dir, f"shard_{idx:05d}.npz")

    with open(input_file, "r") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            moves = encoder.pgn_to_sequence(game)
            moves = np.array(moves, dtype=np.int32)

            shard_game_ids.append(global_game_id)
            shard_tokens.append(moves)
            global_game_id += 1

    # Save final partial shard
    if shard_game_ids:
        save_shard(shard_path(shard_idx), shard_game_ids, shard_tokens)
        shard_idx += 1

    return shard_idx, global_game_id


def run_through_folder(input_folder, output_folder, encoder):
    os.makedirs(output_folder, exist_ok=True)

    shard_idx = 0
    global_game_id = 0

    for input_file in os.listdir(input_folder):
        if not input_file.endswith(".pgn"):
            continue

        print(f"Processing {input_file}...")
        input_path = os.path.join(input_folder, input_file)

        shard_idx, global_game_id = process_pgn_file(
            input_path,
            output_folder,
            encoder,
            shard_start_index=shard_idx,
            global_game_id=global_game_id,
        )

    print(f"Done. Total games: {global_game_id}, total shards: {shard_idx}")
