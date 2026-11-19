"""Extend DET grid with additional seeds (3-5) for cond-on and cond-off.

Appends new rows to existing results JSONs and prints a running summary
combining old + new seeds to decide the detector-mode narrative.

Usage:
    python experiments/run_det_ext.py
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
CONDS = ["on", "off"]
NEW_SEEDS = [3, 4, 5]
EPISODES = 250
EVAL_EPS = 10


def load_rows(path):
    if os.path.exists(path):
        return json.load(open(path)).get("rows", [])
    return []


def save_rows(path, rows):
    with open(path, "w") as f:
        json.dump({"rows": rows}, f, indent=2)


def run_one(cfg, variant, cond, seed):
    run_cfg = copy.deepcopy(cfg)
    run_cfg["seed"] = seed
    run_cfg["rl"]["replay"] = variant
    run_cfg["rl"]["strategy_source"] = "detector"
    run_cfg["rl"]["cond_on_strategy"] = (cond == "on")
    run_cfg["rl"]["episodes"] = EPISODES
    print(f"\n=== [det/{cond}] train {variant} seed={seed} ===")
    t0 = time.time()
    out = train_with_config(run_cfg, silent=False)
    train_time = time.time() - t0
    summary = evaluate_model(run_cfg, out["model_path"], episodes=EVAL_EPS, online=True)
    row = {
        "tagging": "det", "cond": cond, "variant": variant, "seed": seed,
        "win_rate": summary["win_rate"],
        "adapt_time": summary["mean_adaptation_time"],
        "mean_reward": float(sum(r["reward"] for r in summary["results"]) / max(1, len(summary["results"]))),
        "transitions_detected": sum(r["transitions"] for r in summary["results"]),
        "train_time": train_time,
    }
    print(f"  => win_rate={row['win_rate']:.3f} adapt={row['adapt_time']:.1f} "
          f"rew={row['mean_reward']:.1f} trans={row['transitions_detected']} train={train_time:.0f}s")
    return row


def report(rows, label):
    print(f"\n===== {label} (all seeds) =====")
    for variant in VARIANTS:
        vr = [r for r in rows if r["variant"] == variant]
        if not vr:
            continue
        wins = [r["win_rate"] for r in vr]
        print(f"  {variant:14s} n={len(wins):2d}  win={sum(wins)/len(wins):.3f} "
              f"(seeds {[r['seed'] for r in vr]})  adapt={sum(r['adapt_time'] for r in vr)/len(vr):5.1f}")


def main():
    cfg = load_config()
    out_on = os.path.join("results", "grid_det_cond_on.json")
    out_off = os.path.join("results", "grid_det_cond_off.json")

    # carry over existing seeds so summary reflects all
    prev_full = load_rows(os.path.join("results", "grid_full.json"))
    prev_off = load_rows(os.path.join("results", "grid_det_off.json"))
    all_on = [r for r in prev_full if r["tagging"] == "det" and r["cond"] == "on"]
    all_off = [r for r in prev_off if r["cond"] == "off"]
    all_on += [r for r in prev_full if r["tagging"] == "det" and r["cond"] == "off"]
    # prevent dupes
    seen = {(r["cond"], r["variant"], r["seed"]): 1 for r in all_on + all_off}

    for cond in CONDS:
        for variant in VARIANTS:
            for seed in NEW_SEEDS:
                if (cond, variant, seed) in seen:
                    continue
                row = run_one(cfg, variant, cond, seed)
                (all_on if cond == "on" else all_off).append(row)
                save_rows(out_on, all_on)
                save_rows(out_off, all_off)

    report(all_on, "DET/cond-on")
    report(all_off, "DET/cond-off")


if __name__ == "__main__":
    main()
