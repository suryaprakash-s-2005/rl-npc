"""Audit outcome metrics with explicit censoring/exclusion statuses.

The submitted paper averaged adaptation time over every evaluation episode.  A
transcript coding error treated episodes that ended before the switch (no post
window) and episodes with an insufficient pre-switch baseline as measured
adaptation times.  The audit reconstructs the intended rule:

    ok        recovery found after the switch
    censored  switch reached, baseline valid, no recovery before episode end
    no-post   episode ended before the switch
    short-base pre-switch window shorter than ROLL_WIN

Only ``ok`` and ``censored`` enter the adaptation-time mean.
"""

import numpy as np

ROLL_WIN = 20


def adaptation_summary(rewards, switch_steps, roll_win=ROLL_WIN):
    rewards = np.asarray(rewards, dtype=np.float64)
    details = []
    for switch_step in switch_steps:
        switch_step = int(switch_step)
        if switch_step >= len(rewards):
            details.append(
                {"switch_step": switch_step, "status": "no-post", "value": None,
                 "pre_mean": None, "target": None}
            )
            continue
        pre = rewards[max(0, switch_step - roll_win):switch_step]
        if len(pre) < roll_win:
            details.append(
                {"switch_step": switch_step, "status": "short-base", "value": None,
                 "pre_mean": float(pre.mean()) if len(pre) else None,
                 "target": None}
            )
            continue
        target = float(pre.mean())
        found = None
        for t in range(switch_step, len(rewards)):
            win = rewards[max(0, t - roll_win + 1):t + 1]
            if float(win.mean()) >= target:
                found = t - switch_step
                break
        if found is None:
            details.append(
                {"switch_step": switch_step, "status": "censored",
                 "value": len(rewards) - switch_step,
                 "pre_mean": float(pre.mean()), "target": target}
            )
        else:
            details.append(
                {"switch_step": switch_step, "status": "ok", "value": found,
                 "pre_mean": float(pre.mean()), "target": target}
            )

    included = [d for d in details if d["status"] in ("ok", "censored")
                and d["value"] is not None]
    values = [float(d["value"]) for d in included]
    if not details:
        aggregate_status = "no-post"
    elif len(details) == 1:
        aggregate_status = details[0]["status"]
    elif any(d["status"] in ("no-post", "short-base") for d in details):
        aggregate_status = "partial"
    elif all(d["status"] == "ok" for d in details):
        aggregate_status = "ok"
    else:
        aggregate_status = "censored"

    return {
        "status": aggregate_status,
        "adaptation_time": float(np.mean(values)) if values else None,
        "included": bool(values),
        "values": values,
        "details": details,
        "n_ok": sum(1 for d in details if d["status"] == "ok"),
        "n_censored": sum(1 for d in details if d["status"] == "censored"),
        "n_no_post": sum(1 for d in details if d["status"] == "no-post"),
        "n_short_base": sum(1 for d in details if d["status"] == "short-base"),
    }


def evaluate_summary(episodes):
    n = max(1, len(episodes))
    wins = sum(int(e.get("caught", 0)) for e in episodes)
    adapt = [float(e["adaptation_time"]) for e in episodes
             if e.get("adaptation_time") is not None
             and e.get("adaptation_status") in ("ok", "censored")]
    statuses = [e.get("adaptation_status", "no-post") for e in episodes]
    return {
        "episodes": len(episodes),
        "win_rate": wins / n,
        "mean_adaptation_time": float(np.mean(adapt)) if adapt else None,
        "median_adaptation_time": float(np.median(adapt)) if adapt else None,
        "n_adapt_included": len(adapt),
        "n_ok": statuses.count("ok"),
        "n_censored": statuses.count("censored"),
        "n_no_post": statuses.count("no-post"),
        "n_short_base": statuses.count("short-base"),
        "n_partial": statuses.count("partial"),
        "transitions_detected": sum(int(e.get("transitions", 0)) for e in episodes),
    }
