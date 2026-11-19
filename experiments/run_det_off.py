"""Targeted DET/cond-off comparison (partitioned vs single).

Tests the hypothesis that strategy-aware replay helps the NPC adapt when the
personality must be inferred (no label in the observation), mirroring the
strong GT/cond-off result. Bounded episodes to keep runtime manageable.

Usage:
    python experiments/run_det_off.py
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.train import train_with_config
from src.eval import evaluate_model

VARIANTS = ["single", "partitioned"]
SEEDS = [0, 1, 2]
EPISODES = 250
EVAL_EPS = 10


def main():
    cfg = load_config()
    os.makedirs("results", exist_ok=True)
    all_rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["seed"] = seed
            run_cfg["rl"]["replay"] = variant
            run_cfg["rl"]["strategy_source"] = "detector"
            run_cfg["rl"]["cond_on_strategy"] = False
            run_cfg["rl"]["episodes"] = EPISODES

            print(f"\n=== [det/off] train {variant} seed={seed} ===")
            t0 = time.time()
            out = train_with_config(run_cfg, silent=False)
            train_time = time.time() - t0
            summary = evaluate_model(run_cfg, out["model_path"], episodes=EVAL_EPS, online=True)
            row = {
                "tagging": "det", "cond": "off", "variant": variant, "seed": seed,
                "win_rate": summary["win_rate"],
                "adapt_time": summary["mean_adaptation_time"],
                "mean_reward": float(sum(r["reward"] for r in summary["results"]) / max(1, len(summary["results"]))),
                "transitions_detected": sum(r["transitions"] for r in summary["results"]),
                "train_time": train_time,
            }
            print(f"  => win_rate={row['win_rate']:.3f} adapt={row['adapt_time']:.1f} "
                  f"rew={row['mean_reward']:.1f} trans={row['transitions_detected']} train={train_time:.0f}s")
            all_rows.append(row)
            with open("results/grid_det_off.json", "w") as f:
                json.dump({"rows": all_rows}, f, indent=2)

    print("\n=== DET/cond-off SUMMARY ===")
    for variant in VARIANTS:
        vr = [r for r in all_rows if r["variant"] == variant]
        print(f"  {variant:14s} win_rate={sum(r['win_rate'] for r in vr)/len(vr):.3f}  "
              f"adapt_time={sum(r['adapt_time'] for r in vr)/len(vr):5.1f}  "
              f"reward={sum(r['mean_reward'] for r in vr)/len(vr):6.1f}  "
              f"trans={sum(r['transitions_detected'] for r in vr):3d}")


if __name__ == "__main__":
    main()
