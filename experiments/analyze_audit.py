"""Aggregate the audit grid into Table II, F4, F5, and F6 summaries."""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audit.config import AUDIT_ROOT
from src.audit.stats import holm, paired_stats


def load_cells(source="groundtruth"):
    rows = []
    for fp in glob.glob(os.path.join(AUDIT_ROOT, "cells", "*.json")):
        with open(fp) as f:
            r = json.load(f)
        if r.get("label_source") == source:
            rows.append(r)
    return rows


def idx(rows, cond, arm, seed):
    for r in rows:
        if r["cond"] == cond and r["arm"] == arm and r["seed"] == seed:
            return r
    return None


def table2(rows, seeds):
    out = []
    for cond in ("off", "on"):
        for mode in ("offline", "online"):
            single, part = [], []
            for seed in seeds:
                a = idx(rows, cond, "single200k", seed)
                b = idx(rows, cond, "partitioned", seed)
                if a and b:
                    single.append(a[mode]["win_rate"])
                    part.append(b[mode]["win_rate"])
            if len(single) < 2:
                continue
            stats = paired_stats(part, single, margin=0.05)
            out.append({
                "cond": cond, "metric": mode,
                "single_mean": float(np.mean(single)),
                "partitioned_mean": float(np.mean(part)),
                "delta": stats["mean_diff"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                "p_wilcoxon": stats["p_wilcoxon"],
                "p_bootstrap": stats["p_bootstrap"],
                "median_diff": stats["median_diff"],
                "cohen_dz": stats["cohen_dz"],
                "mde_80": stats["mde_80"],
                "tost_0.05": stats.get("tost"),
                "n": stats["n"],
                "seeds": seeds,
                "single": single, "partitioned": part,
            })
    ps = [r["p_wilcoxon"] for r in out]
    for r, ph in zip(out, holm(ps)):
        r["p_holm"] = ph
    return out


def f4(rows, seeds):
    out = []
    for cond in ("off", "on"):
        for mode in ("offline", "online"):
            big, small = [], []
            for seed in seeds:
                a = idx(rows, cond, "single200k", seed)
                b = idx(rows, cond, "single50k", seed)
                if a and b:
                    big.append(a[mode]["win_rate"])
                    small.append(b[mode]["win_rate"])
            if len(big) < 2:
                continue
            stats = paired_stats(small, big, margin=0.05)
            out.append({
                "cond": cond, "metric": mode,
                "single200k_mean": float(np.mean(big)),
                "single50k_mean": float(np.mean(small)),
                "delta_50k_minus_200k": stats["mean_diff"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                "p_wilcoxon": stats["p_wilcoxon"],
                "n": stats["n"],
            })
    return out


def f6_counts(rows):
    counts = {"cells": 0, "episodes": 0, "no_post": 0, "short_base": 0,
              "censored": 0, "ok": 0}
    for r in rows:
        tag = r["tag"]
        for mode in ("offline", "online"):
            fp = os.path.join(AUDIT_ROOT, "eval", tag, f"{mode}.episodes.jsonl")
            if not os.path.exists(fp):
                continue
            counts["cells"] += 1
            with open(fp) as f:
                for line in f:
                    e = json.loads(line)
                    counts["episodes"] += 1
                    st = e.get("adaptation_status")
                    if st == "no-post":
                        counts["no_post"] += 1
                    elif st == "short-base":
                        counts["short_base"] += 1
                    elif st == "censored":
                        counts["censored"] += 1
                    elif st == "ok":
                        counts["ok"] += 1
    return counts


def f5_cell(cfg_tag, arm, cond, seed):
    fp = os.path.join(AUDIT_ROOT, "train", cfg_tag, "episodes.jsonl")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        hist = [json.loads(line) for line in f]
    blocks = []
    for start in range(0, len(hist), 50):
        chunk = hist[start:start + 50]
        if not chunk:
            continue
        blocks.append({
            "start": start,
            "end": start + len(chunk) - 1,
            "catch_rate": float(np.mean([e["caught"] for e in chunk])),
            "q_mean_abs": float(np.mean([e["q_mean_abs"] for e in chunk
                                         if e.get("q_mean_abs") is not None])),
            "q_max_abs": float(np.max([e["q_max_abs"] for e in chunk
                                       if e.get("q_max_abs") is not None])),
        })
    first_q = [e["q_mean_abs"] for e in hist if e.get("q_mean_abs") is not None]
    onset = None
    for e in hist:
        if e.get("q_mean_abs") is not None and e["q_mean_abs"] > 1000:
            onset = e["episode"]
            break
    return {
        "tag": cfg_tag, "arm": arm, "cond": cond, "seed": seed,
        "blocks": blocks,
        "first_block_q_mean": float(np.mean(first_q[:50])) if first_q else None,
        "max_q_mean": float(np.max(first_q)) if first_q else None,
        "onset_q_gt_1000": onset,
        "final_catch_rate": blocks[-1]["catch_rate"] if blocks else None,
    }


def f5(rows):
    out = []
    for cond in ("off", "on"):
        for seed in (0, 5):
            for arm in ("single200k", "partitioned", "single50k"):
                r = idx(rows, cond, arm, seed)
                if not r:
                    continue
                tag = r["tag"]
                cell = f5_cell(tag, arm, cond, seed)
                if cell:
                    out.append(cell)
    return out


def main():
    seeds = [0, 1, 2, 3, 4, 5]
    rows = load_cells("groundtruth")
    print(f"loaded {len(rows)} groundtruth cells")
    result = {
        "n_cells": len(rows),
        "table2": table2(rows, seeds),
        "f4": f4(rows, seeds),
        "f6": f6_counts(rows),
        "f5": f5(rows),
    }
    out = os.path.join(AUDIT_ROOT, "analysis_groundtruth.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("f6", "table2")}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
