"""Player strategy definitions and parameter mapping.

A "strategy" is a parameter vector that configures how the simulated player
(opponent) behaves inside the chase game:

  [aggression, retreat_rate, evasion_distance, max_speed_mult, engagement_dist]

- aggression:       0..1 tendency to approach/harass the NPC
- retreat_rate:     0..1 tendency to back away when the NPC approaches
- evasion_distance: how far the player keeps from the NPC (0 = none)
- max_speed_mult:   speed multiplier relative to base player speed
- engagement_dist:  preferred distance to the NPC for "engaging"

The four classes map to the PPT: aggressive, defensive, evasive, balanced.
"""

from enum import IntEnum


class Strategy(IntEnum):
    AGGRESSIVE = 0
    DEFENSIVE = 1
    EVASIVE = 2
    BALANCED = 3


LABEL_TO_ID = {s.name.lower(): int(s) for s in Strategy}
ID_TO_LABEL = {int(s): s.name.lower() for s in Strategy}


def params_for_label(cfg, label: str):
    """Return the player parameter vector for a strategy label."""
    label = label.lower()
    if label not in cfg["strategy"]["params"]:
        raise KeyError(f"unknown strategy label '{label}'")
    return list(cfg["strategy"]["params"][label])


def strategy_labels(cfg):
    return list(cfg["strategy"]["labels"])


def build_strategy_player(seed=0):
    """A tiny random player controller with strategy-like tendencies.

    This is only used as a fallback/demo when the detector is not the source
    of truth. The real strategy-driven player lives in player.py and is driven
    by an explicit strategy parameter vector.
    """
    import random

    rng = random.Random(seed)
    return {
        "aggression": rng.uniform(0.0, 1.0),
        "evasion_distance": rng.uniform(0.0, 2.0),
        "clinging": rng.uniform(0.0, 1.0),
    }