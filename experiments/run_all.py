"""Run baseline vs strategy-aware experiments and produce a comparison.

Usage:
    python -m experiments.run_all [--seeds 0 1 2] [--variants single partitioned]
        [--tagging gt|det] [--cond on|off] [--episodes 150] [--eval_episodes 12]

Trains each variant across seeds, evaluates each trained model with online
updates and a mid-episode strategy switch, and prints a summary table.

`--tagging det` uses the online classifier + CUSUM transition detector;
`--tagging gt` uses the player's ground-truth strategy (isolates RL effects).
`--cond off` disables policy conditioning on the detected strategy (baseline
obs stays 7-dim); `--cond on` (the default) appends a strategy one-hot to the
state. The "cond" value is recorded per row so a conditioning ablation can be
compared directly.

Results are written to results/run_all_<tagging>.json (plus the legacy
run_all_summary.json for forwards compatibility).
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.train import train_with_config
from src.eval import evaluate_model

VARIANTS = ["single", "partitioned", "partitioned_noprior"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2",
                        help="comma-separated seed list")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--tagging", choices=["gt", "det"], default="gt")
    parser.add_argument("--cond", choices=["on", "off"], default="on")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--eval_episodes", type=int, default=12)
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v for v in args.variants.split(",") if v.strip()]
    return seeds, variants, args.tagging, args.cond, args.episodes, args.eval_episodes


def main():
    cfg = load_config()
    seeds, variants, tagging, cond, episodes, eval_episodes = parse_args(sys.argv[1:])
    if tagging == "det" and not os.path.exists(cfg["detector"]["model_path"]):
        print(f"ERROR detector model not found ({cfg['detector']['model_path']}); "
              f"train it with `python -m src.strategy.train_detector` first")
        sys.exit(1)

    strategy_source = "groundtruth" if tagging == "gt" else "detector"
    cond_on = (cond == "on")
    rows = []
    for variant in variants:
        for seed in seeds:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["seed"] = seed
            run_cfg["rl"]["replay"] = variant
            run_cfg["rl"]["episodes"] = episodes
            run_cfg["rl"]["strategy_source"] = strategy_source  # groundtruth / detector
            run_cfg["rl"]["cond_on_strategy"] = cond_on
            print(f"\n=== [{tagging}/{cond}] train {variant} seed={seed} ===")
            out = train_with_config(run_cfg, silent=False)
            summary = evaluate_model(run_cfg, out["model_path"],
                                     episodes=eval_episodes, online=True)
            rows.append({
                "tagging": tagging, "cond": cond_on, "variant": variant, "seed": seed,
                "mean_reward": float(np.mean([r["reward"] for r in summary["results"]])),
                "win_rate": summary["win_rate"],
                "adapt_time": summary["mean_adaptation_time"],
            })
            print(f"  => win_rate={summary['win_rate']:.2f} "
                  f"adapt_time={summary['mean_adaptation_time']:.1f}")

    # aggregate
    os.makedirs("results", exist_ok=True)
    files = [f"run_all_{tagging}_{cond}.json", f"run_all_{tagging}.json"]
    for fn in files:
        with open(os.path.join("results", fn), "w") as f:
            json.dump({"rows": rows, "tagging": tagging, "cond": cond_on}, f, indent=2)
    with open("results/run_all_summary.json", "w") as f:
        json.dump({"rows": rows, "tagging": tagging, "cond": cond_on}, f, indent=2)

    print(f"\n===== SUMMARY [{tagging}/{cond}] (mean over seeds) =====")
    for variant in variants:
        rr = [r for r in rows if r["variant"] == variant]
        if not rr:
            continue
        print(f"{variant:22s} win_rate={np.mean([r['win_rate'] for r in rr]):.3f} "
              f"adapt_time={np.mean([r['adapt_time'] for r in rr]):5.1f} "
              f"reward={np.mean([r['mean_reward'] for r in rr]):6.2f}")


if __name__ == "__main__":
    main()