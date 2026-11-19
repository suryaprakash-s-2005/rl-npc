"""Run the full experiment grid for the combat adaptation study.

Runs GT (groundtruth) and DET (detector) configurations across seeds,
single vs partitioned replay, and cond-on/off conditions.

Usage:
    python experiments/run_grid.py
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
GT_SEEDS = [0, 1, 2]
DET_SEEDS = [0, 1, 2]
EVAL_EPS = 12


def run_one(cfg, tagging, cond, variant, seed):
    run_cfg = copy.deepcopy(cfg)
    run_cfg["seed"] = seed
    run_cfg["rl"]["replay"] = variant
    run_cfg["rl"]["strategy_source"] = "groundtruth" if tagging == "gt" else "detector"
    run_cfg["rl"]["cond_on_strategy"] = (cond == "on")

    print(f"\n=== [{tagging}/{cond}] train {variant} seed={seed} ===")
    t0 = time.time()
    out = train_with_config(run_cfg, silent=False)
    train_time = time.time() - t0

    summary = evaluate_model(run_cfg, out["model_path"],
                             episodes=EVAL_EPS, online=True)
    row = {
        "tagging": tagging, "cond": cond, "variant": variant, "seed": seed,
        "win_rate": summary["win_rate"],
        "adapt_time": summary["mean_adaptation_time"],
        "mean_reward": float(
            sum(r["reward"] for r in summary["results"]) / max(1, len(summary["results"]))
        ),
        "transitions_detected": sum(r["transitions"] for r in summary["results"]),
        "train_time": train_time,
    }
    print(f"  => win_rate={row['win_rate']:.3f} adapt_time={row['adapt_time']:.1f} "
          f"reward={row['mean_reward']:.1f} transitions={row['transitions_detected']} "
          f"train={train_time:.0f}s")
    return row


def main():
    cfg = load_config()
    os.makedirs("results", exist_ok=True)

    all_rows = []
    t_start = time.time()

    # GT grid: cond-on and cond-off
    for cond in ["on", "off"]:
        for variant in VARIANTS:
            for seed in GT_SEEDS:
                row = run_one(cfg, "gt", cond, variant, seed)
                all_rows.append(row)
                # save after each run
                with open("results/grid_full.json", "w") as f:
                    json.dump({"rows": all_rows}, f, indent=2)

    # DET grid: cond-on only
    for variant in VARIANTS:
        for seed in DET_SEEDS:
            row = run_one(cfg, "det", "on", variant, seed)
            all_rows.append(row)
            with open("results/grid_full.json", "w") as f:
                json.dump({"rows": all_rows}, f, indent=2)

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print(f"FULL GRID SUMMARY (total {total_time:.0f}s)")
    print(f"{'='*70}")
    configs = [
        ("gt", "on", "GT/cond-on"),
        ("gt", "off", "GT/cond-off"),
        ("det", "on", "DET/cond-on"),
    ]
    for tagging, cond, label in configs:
        subset = [r for r in all_rows if r["tagging"] == tagging and r["cond"] == cond]
        if not subset:
            continue
        print(f"\n  {label}:")
        for variant in VARIANTS:
            vr = [r for r in subset if r["variant"] == variant]
            if not vr:
                continue
            print(f"    {variant:14s}  win_rate={sum(r['win_rate'] for r in vr)/len(vr):.3f}  "
                  f"adapt_time={sum(r['adapt_time'] for r in vr)/len(vr):5.1f}  "
                  f"reward={sum(r['mean_reward'] for r in vr)/len(vr):6.1f}  "
                  f"trans={sum(r['transitions_detected'] for r in vr):3d}")

    print(f"\nSaved -> results/grid_full.json")


if __name__ == "__main__":
    main()
