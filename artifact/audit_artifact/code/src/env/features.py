"""Sliding-window gameplay telemetry feature extraction.

From the raw per-step telemetry of the combat game we compute movement and
combat features over a window:

Movement features (as before):
  1. attack_frequency        -- fraction of steps the player approached the NPC
  2. engagement_distance     -- mean distance to the NPC while engaging
  3. retreat_frequency       -- fraction of steps the player retreated from NPC
  4. defensive_action_ratio  -- retreats / (approaches + retreats)
  5. engagement_duration     -- longest run of consecutive engagement steps / W

Combat features (new):
  6. combat_attack_freq      -- fraction of steps the player swung its sword
  7. hit_rate                -- fraction of player attacks that landed
  8. damage_dealt_rate       -- damage dealt to the NPC / W
  9. in_range_fraction       -- fraction of steps the NPC was in sword range
 10. low_health_retreat      -- fraction of low-health steps where the player
                               retreated (defensive personality backs off when hurt)

Together these form the input vector for the strategy classifier.
"""

import collections

import numpy as np


ENGAGEMENT_BAND = 3.0  # distance below which a player is 'engaging'


class FeatureWindow:
    """Maintains a rolling window of raw step telemetry."""

    def __init__(self, window: int):
        self.window = window
        self._raw = collections.deque(maxlen=window)
        self._n_steps = 0

    def reset(self):
        self._raw.clear()
        self._n_steps = 0

    @staticmethod
    def extract_step(npc_pos, prev_npc_pos, player_pos, prev_player_pos,
                     combat=None):
        """Raw per-step telemetry derived from positions and combat events.

        `combat` (optional dict) may carry:
          player_attack : bool   player swung its sword this step
          player_hit    : bool   that swing landed (dealt damage)
          damage_dealt  : float  damage dealt to the NPC this step
          player_hp_frac: float  0..1 player remaining health
        """
        dist = float(np.linalg.norm(player_pos - npc_pos))
        d_prev = float(np.linalg.norm(prev_player_pos - prev_npc_pos))

        closing_rate = d_prev - dist

        approach = closing_rate > 0.08
        retreat = closing_rate < -0.08
        side_move = not (approach or retreat)

        engagement = dist < ENGAGEMENT_BAND

        c = combat or {}
        player_attack = bool(c.get("player_attack", False))
        player_hit = bool(c.get("player_hit", False))
        damage_dealt = float(c.get("damage_dealt", 0.0))
        hp_frac = c.get("player_hp_frac")
        low_health = (hp_frac is not None and hp_frac < 0.4)

        return {
            "dist": dist,
            "approach": approach,
            "retreat": retreat,
            "side_move": side_move,
            "engagement": engagement,
            "attack": player_attack,
            "hit": player_hit,
            "damage_dealt": damage_dealt,
            "low_health": low_health,
        }

    def update(self, raw):
        self._raw.append(raw)
        self._n_steps += 1

    # ------------------------------------------------------------------ #
    def features(self):
        """Return the 10-feature sliding-window vector (or None if short)."""
        n = len(self._raw)
        if n < max(4, self.window // 2):  # need a minimum history
            return None
        w = self._raw
        approaches = sum(1 for r in w if r["approach"])
        retreats = sum(1 for r in w if r["retreat"])
        dists = [r["dist"] for r in w]

        attack_freq = approaches / n
        retreat_freq = retreats / n
        denom = approaches + retreats
        def_ratio = retreats / denom if denom > 0 else 0.0

        best = cur = 0
        for r in w:
            if r["engagement"]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        engage_dur = best / n

        # combat features
        swings = sum(1 for r in w if r["attack"])
        hits = sum(1 for r in w if r["hit"])
        combat_attack_freq = swings / n
        hit_rate = hits / max(swings, 1)
        damage_dealt_rate = sum(r["damage_dealt"] for r in w) / n
        in_range = sum(1 for r in w if r["dist"] < 1.8) / n
        low_hp_steps = sum(1 for r in w if r["low_health"])
        low_hp_retreats = sum(
            1 for r in w if r["low_health"] and r["retreat"]
        )
        low_health_retreat = low_hp_retreats / max(low_hp_steps, 1)

        return np.array(
            [
                attack_freq,
                sum(dists) / n,
                retreat_freq,
                def_ratio,
                engage_dur,
                combat_attack_freq,
                hit_rate,
                damage_dealt_rate,
                in_range,
                low_health_retreat,
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------ #
    def is_ready(self):
        return self.features() is not None


def feature_dim():
    return 10
