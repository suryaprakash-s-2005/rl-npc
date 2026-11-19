"""Run the audit grid (train + offline/online evaluation) with resumable cells."""

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import ARMS, AUDIT_ROOT, audit_config
from src.audit.runner import evaluate_audit, train_audit

CELL_DIR = os.path.join(AUDIT_ROOT, "cells")


def run_cell(spec):
    tag = spec["tag"]
    cell_path = os.path.join(CELL_DIR, f"{tag}.json")
    if os.path.exists(cell_path):
        with open(cell_path) as f:
            return json.load(f)
    cfg = audit_config(
        seed=spec["seed"], arm=spec["arm"],
        label_source=spec["label_source"], cond=spec["cond"],
        episodes=spec["episodes"], eval_episodes=spec["eval_episodes"],
        switch_steps=tuple(spec.get("switch_steps", [40])),
        episode_len=spec.get("episode_len", 400),
        network=spec.get("network", "dqn"),
        tag=tag,
    )
    t0 = time.time()
    out = train_audit(cfg, silent=True)
    offline = evaluate_audit(cfg, out["model_path"], mode="offline", silent=True)
    online = evaluate_audit(cfg, out["model_path"], mode="online", silent=True)
    keys = ("win_rate", "mean_adaptation_time", "median_adaptation_time",
            "n_adapt_included", "n_ok", "n_censored", "n_no_post",
            "n_short_base")
    row = {
        "tag": tag, "arm": spec["arm"],
        "cond": "on" if spec["cond"] else "off",
        "label_source": spec["label_source"], "seed": spec["seed"],
        "checkpoint": out["model_path"],
        "offline": {k: offline[k] for k in keys},
        "online": {k: online[k] for k in keys},
        "train_elapsed_s": out["summary"]["elapsed_s"],
        "wall_elapsed_s": time.time() - t0,
    }
    os.makedirs(CELL_DIR, exist_ok=True)
    with open(cell_path, "w") as f:
        json.dump(row, f, indent=2)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-source", choices=["groundtruth", "detector"],
                        default="groundtruth")
    parser.add_argument("--seeds", default="0,1,2,3,4,5")
    parser.add_argument("--arms", default="single200k,partitioned,single50k")
    parser.add_argument("--conditions", default="off,on")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--switch-steps", default="40")
    parser.add_argument("--episode-len", type=int, default=400)
    parser.add_argument("--network", default="dqn")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tag-prefix", default="")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a for a in args.arms.split(",") if a.strip()]
    conds = [c == "on" for c in args.conditions.split(",") if c.strip()]
    switch_steps = [int(s) for s in args.switch_steps.split(",") if s.strip()]

    specs = []
    for cond in conds:
        for arm in arms:
            if arm not in ARMS:
                raise KeyError(arm)
            for seed in seeds:
                base = audit_config(
                    seed=seed, arm=arm, label_source=args.label_source,
                    cond=cond, episodes=args.episodes,
                    eval_episodes=args.eval_episodes,
                    switch_steps=tuple(switch_steps),
                    episode_len=args.episode_len,
                    network=args.network,
                )
                tag = base["audit"]["tag"]
                if args.tag_prefix:
                    tag = f"{args.tag_prefix}-{tag}"
                specs.append({
                    "tag": tag, "arm": arm, "cond": cond,
                    "label_source": args.label_source, "seed": seed,
                    "episodes": args.episodes,
                    "eval_episodes": args.eval_episodes,
                    "switch_steps": switch_steps,
                    "episode_len": args.episode_len,
                    "network": args.network,
                })

    os.makedirs(CELL_DIR, exist_ok=True)
    rows = []
    todo = [s for s in specs if not os.path.exists(
        os.path.join(CELL_DIR, f"{s['tag']}.json"))]
    print(f"grid: {len(specs)} cells, {len(todo)} remaining, workers={args.workers}",
          flush=True)
    if not todo:
        for s in specs:
            with open(os.path.join(CELL_DIR, f"{s['tag']}.json")) as f:
                rows.append(json.load(f))
    else:
        with futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(run_cell, s): s for s in todo}
            for fut in futures.as_completed(futs):
                spec = futs[fut]
                try:
                    row = fut.result()
                    rows.append(row)
                    print(f"done {spec['tag']} win_off={row['offline']['win_rate']:.3f} "
                          f"win_onl={row['online']['win_rate']:.3f}", flush=True)
                except Exception as exc:
                    print(f"FAILED {spec['tag']}: {exc!r}", flush=True)
                out = os.path.join(AUDIT_ROOT, f"grid_{args.label_source}.json")
                with open(out, "w") as f:
                    json.dump({"rows": rows}, f, indent=2)
        # fill skipped rows
        for s in specs:
            fp = os.path.join(CELL_DIR, f"{s['tag']}.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    r = json.load(f)
                if not any(x["tag"] == r["tag"] for x in rows):
                    rows.append(r)

    out = os.path.join(AUDIT_ROOT, f"grid_{args.label_source}.json")
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"saved {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
