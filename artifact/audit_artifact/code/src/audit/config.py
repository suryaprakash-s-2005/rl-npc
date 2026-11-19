"""Audit protocol configuration.

The submitted audit protocol was: player HP 60, NPC HP 20, switch at step 40,
400 training episodes, 12 evaluation episodes, 6 paired seeds, and three replay
arms (single 200k, partitioned, single 50k).

This module builds those configurations without requiring PyYAML and keeps the
original defaults untouched.
"""

import copy
import os

from ..defaults import DEFAULTS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT_ROOT = os.path.join(PROJECT_ROOT, "results", "audit")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "audit")

LABELS = ["aggressive", "defensive", "evasive", "balanced"]
SWITCH_PAIRS = [
    ("aggressive", "defensive"),
    ("defensive", "aggressive"),
    ("evasive", "balanced"),
    ("balanced", "evasive"),
]

ARMS = {
    "single200k": {"replay": "single", "buffer_capacity": 200000},
    # capacity is per partition; four partitions therefore hold 200,000 total,
    # matching the single-200k arm's total capacity.
    "partitioned": {"replay": "partitioned", "buffer_capacity": 50000},
    "single50k": {"replay": "single", "buffer_capacity": 50000},
}


def merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def audit_config(
    seed=0,
    arm="partitioned",
    label_source="groundtruth",
    cond=False,
    episodes=400,
    eval_episodes=12,
    switch_steps=(40,),
    episode_len=400,
    world_size=20.0,
    network="dqn",
    weight_decay=0.0,
    target_clip=0.0,
    mask_oracle_onehot=False,
    legacy_global_rng=False,
    replay_seed_offset=0,
    f5_trace=False,
    params_jitter=0.2,
    detector_path=None,
    tag=None,
    **overrides,
):
    """Build one audit run configuration.

    Parameters mirror the paper protocol.  ``switch_steps`` may contain more
    than one step for the multi-switch robustness condition.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown audit arm '{arm}'")
    arm_cfg = ARMS[arm]

    cfg = merge(
        DEFAULTS,
        {
            "seed": int(seed),
            "env": {
                "world_size": float(world_size),
                "episode_len": int(episode_len),
            },
            "combat": {
                "player_hp": 60,
                "npc_hp": 20,
            },
            "rl": {
                "replay": arm_cfg["replay"],
                "buffer_capacity": int(arm_cfg["buffer_capacity"]),
                "episodes": int(episodes),
                "strategy_source": label_source,
                "cond_on_strategy": bool(cond),
                "in_episode_switch": True,
                "switch_step": int(switch_steps[0]) if switch_steps else 0,
                "network": network,
                "weight_decay": float(weight_decay),
                "target_clip": float(target_clip),
                "mask_oracle_onehot": bool(mask_oracle_onehot),
                "legacy_global_rng": bool(legacy_global_rng),
                "replay_seed_offset": int(replay_seed_offset),
                "f5_trace": bool(f5_trace),
                "params_jitter": float(params_jitter),
                "seed_demos": int(DEFAULTS["rl"].get("seed_demos", 30)),
            },
            "detector": {
                "model_path": detector_path
                or os.path.join(DATA_ROOT, "detector_hp60_randomwalk.pt"),
            },
            "audit": {
                "arm": arm,
                "label_source": label_source,
                "cond": bool(cond),
                "switch_steps": [int(s) for s in switch_steps],
                "eval_episodes": int(eval_episodes),
                "roll_win": 20,
                "detector_label": "randomwalk",
                "protocol": "hp60-npc20-switch40",
                "tag": tag,
            },
        },
    )
    cfg = merge(cfg, overrides)
    if not cfg["audit"].get("tag"):
        cfg["audit"]["tag"] = make_tag(cfg)
    return cfg


def make_tag(cfg):
    audit = cfg["audit"]
    cond = "on" if audit["cond"] else "off"
    src = "gt" if audit["label_source"] == "groundtruth" else "det"
    switches = "-".join(str(s) for s in audit["switch_steps"]) or "none"
    return (
        f"{audit['arm']}-{src}-{cond}-ep{int(cfg['rl']['episodes'])}"
        f"-sw{switches}-seed{int(cfg['seed'])}"
    )


def run_dir(cfg, kind="train"):
    return os.path.join(AUDIT_ROOT, kind, cfg["audit"]["tag"])


def checkpoint_path(cfg):
    return os.path.join(DATA_ROOT, "checkpoints", f"{cfg['audit']['tag']}.pt")
