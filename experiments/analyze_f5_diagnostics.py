"""F5 diagnostics from the corrected per-episode training records.

No training is performed.  The script localizes the critic scale divergence and
the catch-rate collapse for the corrected pairwise seed-2 case and the controls.

Run from the project root:
    python -m experiments.analyze_f5_diagnostics
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "results", "audit", "train")
OUT = os.path.join(ROOT, "results", "audit", "f5_diagnostics.json")

ARMS = ("single200k", "single50k", "partitioned")
CONDS = ("on", "off")
SEEDS = (2, 5)


def load_episodes(arm, cond, seed):
    tag = f"{arm}-gt-{cond}-ep400-sw40-seed{seed}"
    path = os.path.join(TRAIN, tag, "episodes.jsonl")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    if not rows:
        return None
    q = [float(r["q_mean_abs"]) for r in rows]
    caught = [int(r["caught"]) for r in rows]
    n = len(rows)
    first50 = caught[:50]
    last50 = caught[-50:]

    def first_above(th):
        for r, value in zip(rows, q):
            if value > th:
                return {
                    "episode": int(r["episode"]),
                    "q_mean_abs": value,
                    "catch_rate_first_50": sum(first50) / max(1, len(first50)),
                }
        return None

    return {
        "n_episodes": n,
        "first_50_catch_rate": sum(first50) / max(1, len(first50)),
        "last_50_catch_rate": sum(last50) / max(1, len(last50)),
        "final_catch_rate": sum(last50) / max(1, len(last50)),
        "q_first": q[0],
        "q_max": max(q),
        "q_last": q[-1],
        "final_caught": caught[-1],
        "onset_episodes": {
            str(th): first_above(th) for th in (1_000, 10_000, 100_000)
        },
    }


def main():
    out = {}
    for cond in CONDS:
        for seed in SEEDS:
            for arm in ARMS:
                rows = load_episodes(arm, cond, seed)
                if rows is None:
                    continue
                out[f"{cond}-seed{seed}-{arm}"] = summarize(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
