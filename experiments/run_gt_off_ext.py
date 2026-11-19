"""Extend GT/cond-off grid with seeds 3-5 (both variants).

Strengthens the headline claim (partitioned > single when the personality must
be inferred) from n=3 to n=6 seeds.

Usage:
    python experiments/run_gt_off_ext.py
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
NEW_SEEDS = [3, 4, 5]
EVAL_EPS = 12


def load_rows(path):
    if os.path.exists(path):
        return json.load(open(path)).get("rows", [])
    return []


def save_rows(path, rows):
    with open(path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)


def main():
    cfg = load_config()
    out = os.path.join("results", "grid_gt_cond_off.json")

    prev_full = load_rows(os.path.join("results", "grid_full.json"))
    all_rows = [r for r in prev_full if r["tagging"] == "gt" and r["cond"] == "off"]
    seen = {(r["variant"], r["seed"]): 1 for r in all_rows}

    for variant in VARIANTS:
        for seed in NEW_SEEDS:
            if (variant, seed) in seen:
                continue
            run_cfg = copy.deepcopy(cfg)
            run_cfg["seed"] = seed
            run_cfg["rl"]["replay"] = variant
            run_cfg["rl"]["strategy_source"] = "groundtruth"
            run_cfg["rl"]["cond_on_strategy"] = False
            print(f"\n=== [gt/off] train {variant} seed={seed} ===")
            t0 = time.time()
            out_m = train_with_config(run_cfg, silent=False)
            train_time = time.time() - t0
            summary = evaluate_model(run_cfg, out_m["model_path"], episodes=EVAL_EPS, online=True)
            row = {
                "tagging": "gt", "cond": "off", "variant": variant, "seed": seed,
                "win_rate": summary["win_rate"],
                "adapt_time": summary["mean_adaptation_time"],
                "mean_reward": float(sum(r["reward"] for r in summary["results"]) / max(1, len(summary["results"]))),
                "transitions_detected": sum(r["transitions"] for r in summary["results"]),
                "train_time": train_time,
            }
            print(f"  => win_rate={row['win_rate']:.3f} adapt={row['adapt_time']:.1f} "
                  f"rew={row['mean_reward']:.1f} train={train_time:.0f}s")
            all_rows.append(row)
            save_rows(out, all_rows)

    print("\n===== GT/cond-off (all seeds) =====")
    for variant in VARIANTS:
        vr = [r for r in all_rows if r["variant"] == variant]
        wins = [r["win_rate"] for r in vr]
        print(f"  {variant:14s} n={len(wins)}  win={sum(wins)/len(wins):.3f} "
              f"(seeds {[r['seed'] for r in vr]})  adapt={sum(r['adapt_time'] for r in vr)/len(vr):5.1f}")


if __name__ == "__main__":
    main()
