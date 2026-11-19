"""Strategy-parameterized player (opponent) controller for the combat game.

The player is a moving target whose movement AND attack behavior is driven by
a strategy parameter vector from strategy.py:

  [aggression, retreat_rate, evasion_distance, max_speed_mult, engagement_dist,
   attack_frequency, attack_range, retreat_health_threshold]

- aggression: 0..1 tendency to approach/harass the NPC
- retreat_rate: 0..1 tendency to back away when the NPC approaches
- evasion_distance: how strongly the player sidesteps (perpendicular) motion
- max_speed_mult: speed multiplier relative to base player speed
- engagement_dist: preferred distance to the NPC for "engaging"
- attack_frequency: 0..1 probability per step of attempting an attack
                   (when in range, facing the NPC, cooldown ready)
- attack_range: reach of the player's sword (world units)
- retreat_health_threshold: below this health fraction the player retreats
                    (defensive personality backs off when hurt)

`set_strategy(label)` switches the behavior at any time, which is exactly the
scenario the project studies: the NPC must detect the switch and adapt.
"""

import numpy as np

from .strategy import params_for_label


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


class StrategyPlayer:
    BASE_SPEED = 2.0

    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.combat = cfg["combat"]
        self.w = float(self.env_cfg["world_size"])
        self.dt = float(self.env_cfg["dt"])
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pos = np.array([self.w / 2.0, self.w / 2.0], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.max_hp = float(self.combat["player_hp"])
        self.hp = self.max_hp
        self.attack_cooldown = 0.0
        self.facing = np.array([1.0, 0.0], dtype=np.float64)
        self._params = np.array([0.5, 0.3, 0.3, 1.0, 2.0, 0.5, 1.5, 0.3])
        self.label = "balanced"
        self._manual = None  # manual velocity override (for human-play demo)
        self._manual_attack = False
        self._manual_aim = None  # aim direction while attacking (mouse demo)

    # ------------------------------------------------------------------ #
    def reset(self, label, pos=None, params_jitter=0.0):
        self.set_strategy(label)
        if params_jitter > 0.0 and self.label in ("aggressive", "defensive", "evasive", "balanced"):
            raw = np.asarray(params_for_label(self.cfg, self.label), dtype=np.float64)
            noise = self.rng.uniform(-params_jitter, params_jitter, size=raw.shape)
            j = raw + noise
            j[3] = _clip(j[3], 0.5, 1.5)        # speed_mult stays sane
            j[4] = _clip(j[4], 0.5, 30.0)       # engagement_dist
            j[6] = _clip(j[6], 1.0, 2.5)        # attack_range
            j[0] = _clip(j[0], 0.0, 1.0)        # aggression
            j[1] = _clip(j[1], 0.0, 1.0)        # retreat_rate
            j[2] = _clip(j[2], 0.0, 3.0)        # evasion_distance
            j[5] = _clip(j[5], 0.0, 1.0)        # attack_frequency
            j[7] = _clip(j[7], 0.0, 1.0)        # retreat_health_threshold
            self._params = j
        if pos is None:
            self.pos = self.rng.uniform(
                self.w * 0.2, self.w * 0.8, size=2
            ).astype(np.float64)
        else:
            self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.hp = self.max_hp
        self.attack_cooldown = 0.0
        return self.pos.copy()

    def set_strategy(self, label):
        self.label = label.lower()
        self._params = np.array(params_for_label(self.cfg, self.label))

    @property
    def aggression(self):
        return self._params[0]

    @property
    def retreat_rate(self):
        return self._params[1]

    @property
    def evasion_distance(self):
        return self._params[2]

    @property
    def speed_mult(self):
        return self._params[3]

    @property
    def engagement_dist(self):
        return self._params[4]

    @property
    def attack_frequency(self):
        return self._params[5]

    @property
    def attack_range(self):
        return self._params[6]

    @property
    def retreat_health_threshold(self):
        return self._params[7]

    # ------------------------------------------------------------------ #
    def _steer(self, npc_pos):
        """Compute a desired unit velocity from the strategy vector."""
        to_npc = npc_pos - self.pos
        dist = float(np.linalg.norm(to_npc))
        if dist < 1e-6:
            toward = np.zeros(2)
        else:
            toward = to_npc / dist
        away = -toward

        # low-health retreat overrides (defensive personality backs off)
        hurt = self.hp / self.max_hp < self.retreat_health_threshold

        approach = self.aggression * toward
        retreat = self.retreat_rate * away
        if hurt:
            approach = np.zeros(2)
            retreat = away  # retreat dominates when hurt

        # Engagement-distance homeostasis (strong, dominant spring):
        engage = np.zeros(2)
        if self.engagement_dist > 0:
            err = (dist - self.engagement_dist) / max(self.engagement_dist, 1e-3)
            engage = _clip(err, -1.0, 1.0) * toward * 1.4

        # Evasion sidestep (perpendicular jitter scaled by evasion_distance)
        evade = np.zeros(2)
        if self.evasion_distance > 1e-3 and dist < self.evasion_distance * 8.0:
            perp = np.array([-toward[1], toward[0]]) if dist > 1e-6 else self.rng.normal(size=2)
            evade = perp * self.evasion_distance * 0.5

        raw = approach + retreat + engage + evade
        norm = float(np.linalg.norm(raw))
        if norm < 1e-6:
            return np.zeros(2)
        speed = min(self.speed_mult * self.BASE_SPEED, 3.0)
        return raw / norm * speed

    def _attack_decision(self, npc_pos, npc_state=None):
        """Decide whether to swing the sword this step (personality-driven)."""
        to_npc = npc_pos - self.pos
        dist = float(np.linalg.norm(to_npc))
        if dist < 1e-6:
            toward = np.zeros(2)
            return False
        toward = to_npc / dist

        in_range = dist < self.attack_range
        cooldown_ok = self.attack_cooldown <= 0.0

        # facing check: facing roughly toward the NPC
        fn = np.linalg.norm(self.facing)
        facing_toward = float(np.dot(self.facing / (fn + 1e-9), toward)) if fn > 1e-6 else 0.0
        aligned = facing_toward > 0.15

        # aggression gating: only attack from a personality that wants to
        want = self.rng.random() < self.attack_frequency
        return bool(in_range and cooldown_ok and aligned and want)

    def step(self, npc_pos, npc_state=None):
        npc_pos = np.asarray(npc_pos, dtype=np.float64)
        if self._manual is not None:
            target = self._manual
            attack = self._manual_attack
        else:
            target = self._steer(npc_pos)
            attack = self._attack_decision(npc_pos, npc_state)

        self.vel = target
        sp = float(np.linalg.norm(target))
        if attack and self._manual_aim is not None:
            # aim-based swing: face the mouse cursor (hit cone follows aim)
            aim = self._manual_aim
            an = float(np.linalg.norm(aim))
            if an > 1e-6:
                self.facing = aim / an
        elif sp > 1e-4:
            self.facing = target / sp
        elif attack:
            # stay-facing: aim at the opponent for the swing
            to_npc = npc_pos - self.pos
            d = float(np.linalg.norm(to_npc))
            if d > 1e-6:
                self.facing = to_npc / d

        self.pos = self.pos + self.vel * self.dt
        self.pos = np.clip(self.pos, 0.0, self.w)

        # (cooldown is ticked once per step by the env's attack resolution;
        #  doing it here too would halve the real swing cadence)

        return self.pos.copy(), attack, self.facing.copy()

    def reset_attack(self):
        self.attack_cooldown = 0.0

    # ------------------------------------------------------------------ #
    def set_manual_velocity(self, vel, attack=False, aim=None):
        """Force the player's velocity (used by the interactive demo).

        `aim`: optional unit/world direction to face while attacking. When
        attacking with an aim, the fighter faces the aim (mouse cursor) so the
        swing's hit cone is resolved against that direction instead of the NPC.
        """
        v = np.asarray(vel, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm > 1e-6 and norm > self.BASE_SPEED * self.speed_mult:
            v = v / norm * (self.BASE_SPEED * self.speed_mult)
        self._manual = v
        self._manual_attack = bool(attack)
        self._manual_aim = None if aim is None else np.asarray(aim, dtype=np.float64)

    def clear_manual(self):
        self._manual = None
        self._manual_attack = False
        self._manual_aim = None

    @property
    def _obs(self):
        return self.pos.copy()
