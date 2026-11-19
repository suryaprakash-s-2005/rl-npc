"""HP calibration sweep for the audit.

Trains a short policy at each player-HP setting and records the offline win
rate and switch reachability.  This is a calibration check, not part of the
deciding grid.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import AUDIT_ROOT, audit_config
from src.audit.runner import evaluate_audit, train_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hps", default="20,40,60,80")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--arm", default="single200k")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--seed-demos", type=int, default=10)
    args = parser.parse_args()

    hps = [int(x) for x in args.hps.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = os.path.join(AUDIT_ROOT, "calibration")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for hp in hps:
        for seed in seeds:
            tag = f"cal-hp{hp}-{args.arm}-seed{seed}"
            cfg = audit_config(
                seed=seed, arm=args.arm, label_source="groundtruth", cond=False,
                episodes=args.episodes, eval_episodes=args.eval_episodes,
                switch_steps=(40,), episode_len=400, tag=tag,
                combat={"player_hp": hp},
            )
            cfg["rl"]["seed_demos"] = args.seed_demos
            out = train_audit(cfg, silent=True)
            off = evaluate_audit(cfg, out["model_path"], mode="offline", silent=True)
            row = {
                "hp": hp, "seed": seed, "arm": args.arm, "tag": tag,
                "offline_win_rate": off["win_rate"],
                "offline_adapt_mean": off["mean_adaptation_time"],
                "offline_adapt_median": off["median_adaptation_time"],
                "switch_reached": off["episodes"] - off["n_no_post"],
                "eval_episodes": off["episodes"],
                "n_no_post": off["n_no_post"],
                "n_short_base": off["n_short_base"],
                "n_censored": off["n_censored"],
                "checkpoint": out["model_path"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            with open(os.path.join(out_dir, "calibration.json"), "w") as f:
                json.dump({"rows": rows}, f, indent=2)
    print(f"saved {os.path.join(out_dir, 'calibration.json')}")


if __name__ == "__main__":
    main()
