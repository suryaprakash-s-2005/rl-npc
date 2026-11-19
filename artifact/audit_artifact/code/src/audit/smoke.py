"""Fast smoke checks for the audit reconstruction.

Run from the project root:
    python -m src.audit.smoke
"""

import json
import os

import numpy as np

from .config import audit_config
from .metrics import adaptation_summary
from .runner import seed_everything, train_audit
from .stats import paired_stats


def check_metrics():
    rewards = [1.0] * 40 + [-5.0] * 5 + [1.0] * 20
    s = adaptation_summary(rewards, switch_steps=(40,), roll_win=20)
    assert s["status"] == "ok", s
    assert s["adaptation_time"] is not None

    short = adaptation_summary([1.0] * 50, switch_steps=(10,), roll_win=20)
    assert short["status"] == "short-base", short

    no_post = adaptation_summary([1.0] * 30, switch_steps=(40,), roll_win=20)
    assert no_post["status"] == "no-post", no_post
    print("metrics: ok")


def check_stats():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a - 1.0
    s = paired_stats(a, b, margin=2.0)
    assert s["mean_diff"] == 1.0
    assert s["ci_low"] <= s["mean_diff"] <= s["ci_high"]
    assert s["wilcoxon_method"].startswith("exact-signflip")
    assert s["tost"]["equivalent"] is True
    rev = paired_stats(b, a)
    assert abs(rev["p_wilcoxon"] - s["p_wilcoxon"]) < 1e-12, rev
    assert abs(rev["p_wilcoxon"] - 0.03125) < 1e-12, rev
    print("stats: ok", json.dumps({k: s[k] for k in
          ("mean_diff", "ci_low", "ci_high", "p_wilcoxon", "mde_80")}, default=str))


def check_determinism():
    runs = []
    for tag in ("smoke-a", "smoke-b"):
        cfg = audit_config(
            seed=0, arm="single50k", label_source="groundtruth", cond=False,
            episodes=2, eval_episodes=1, switch_steps=(20,), episode_len=60,
            params_jitter=0.0, f5_trace=True, tag=tag,
        )
        cfg["rl"]["seed_demos"] = 0
        cfg["rl"]["batch_size"] = 8
        out = train_audit(cfg, silent=True)
        runs.append(out["history"])
    if runs[0] != runs[1]:
        for a, b in zip(runs[0], runs[1]):
            if a != b:
                print("first differing episode:", a, b)
                break
        raise AssertionError("same-seed training runs are not identical")
    print("determinism: ok")


def main():
    seed_everything(0)
    check_metrics()
    check_stats()
    check_determinism()
    print("all smoke checks passed")


if __name__ == "__main__":
    main()
