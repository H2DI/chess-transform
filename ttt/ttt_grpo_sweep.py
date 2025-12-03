from itertools import product
import subprocess
import json

# define search space
sweep = {
    "learning_rate": [1e-5, 1e-4],
    "beta": [0.01, 0.2],
    "group_size": [32, 128],
}


def grid(sweep):
    keys = sweep.keys()
    vals = sweep.values()
    for i, combo in enumerate(product(*vals)):
        r = dict(zip(keys, combo))
        r["session_name"] = f"run_{i}"
        yield r


if __name__ == "__main__":
    for i, params in enumerate(grid(sweep)):
        config_path = f"configs/run_{i}.json"
        with open(config_path, "w") as f:
            json.dump(params, f)

        subprocess.Popen(
            ["python", "-u", "ttt_rl_train.py", "--config", config_path],
            stdout=open(f"logs/run_{i}.out", "w"),
            stderr=open(f"logs/run_{i}.err", "w"),
        )
