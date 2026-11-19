"""Run one audit cell (train + offline/online evaluation)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import audit_config, AUDIT_ROOT
from src.audit.runner import evaluate_audit, train_audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="single200k")
    parser.add_argument("--cond", choices=["on", "off"], default="off")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-source", choices=["groundtruth", "detector"],
                        default="groundtruth")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--switch-steps", default="40")
    parser.add_argument("--episode-len", type=int, default=400)
    parser.add_argument("--network", default="dqn")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    cfg = audit_config(
        seed=args.seed,
        arm=args.arm,
        label_source=args.label_source,
        cond=(args.cond == "on"),
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        switch_steps=tuple(int(s) for s in args.switch_steps.split(",") if s),
        episode_len=args.episode_len,
        network=args.network,
        tag=args.tag,
    )
    out = train_audit(cfg, silent=args.silent)
    offline = evaluate_audit(cfg, out["model_path"], mode="offline",
                             silent=args.silent)
    online = evaluate_audit(cfg, out["model_path"], mode="online",
                            silent=args.silent)
    row = {
        "tag": cfg["audit"]["tag"],
        "arm": cfg["audit"]["arm"],
        "cond": "on" if cfg["audit"]["cond"] else "off",
        "label_source": cfg["audit"]["label_source"],
        "seed": cfg["seed"],
        "checkpoint": out["model_path"],
        "offline": {k: offline[k] for k in
                    ("win_rate", "mean_adaptation_time", "median_adaptation_time",
                     "n_adapt_included", "n_ok", "n_censored", "n_no_post",
                     "n_short_base")},
        "online": {k: online[k] for k in
                   ("win_rate", "mean_adaptation_time", "median_adaptation_time",
                    "n_adapt_included", "n_ok", "n_censored", "n_no_post",
                    "n_short_base")},
        "train_elapsed_s": out["summary"]["elapsed_s"],
    }
    out_dir = os.path.join(AUDIT_ROOT, "cells")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{cfg['audit']['tag']}.json"), "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
