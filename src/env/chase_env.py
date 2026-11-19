"""Gymnasium-style symmetric combat environment.

The NPC (the trained agent) must defeat the player by reducing its HP to 0.
The player is a strategy-driven opponent that can switch strategies
mid-episode - the scenario studied in the project. The NPC must also keep its
own HP above 0 (symmetric combat).

Action space : Discrete(18)  ->  9 movement directions x {move, attack}.
              Index a: move_dir = a % 9 (stay + 8 compass directions),
              is_attack = a >= 9.
Observation  : 14-dim vector (relative pos/vel, healths, cooldowns,
              engagement flags, heading, closing signal).
"""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None

from .player import StrategyPlayer
from .strategy import LABEL_TO_ID

N_ACTIONS = 18
_DIAG = 0.7071
_DIRS = [
    np.array([0.0, 0.0]),
    np.array([1.0, 0.0]),
    np.array([-1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([0.0, -1.0]),
    np.array([_DIAG, _DIAG]),
    np.array([_DIAG, -_DIAG]),
    np.array([-_DIAG, _DIAG]),
    np.array([-_DIAG, -_DIAG]),
]


def move_dir_of(action):
    return int(action) % len(_DIRS)


def is_attack(action):
    return int(action) >= len(_DIRS)


class CombatEntity:
    """Small holder for combat state shared by player and NPC."""

    __slots__ = ("max_hp", "hp", "attack_cooldown", "facing", "pos", "vel")

    def __init__(self, max_hp, pos, facing):
        self.max_hp = float(max_hp)
        self.hp = self.max_hp
        self.attack_cooldown = 0.0
        self.facing = np.asarray(facing, dtype=np.float64)
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)


class ChaseEnv:
    """Lightweight env not subclassing gym.Env to keep deps optional."""

    metadata = {"render_modes": []}

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.combat = cfg["combat"]
        self.w = float(self.env_cfg["world_size"])
        self.dt = float(self.env_cfg["dt"])
        self.npc_max_speed = float(self.env_cfg["npc_max_speed"])
        self.sensor_range = float(self.env_cfg["npc_sensor_range"])
        self.episode_len = int(self.env_cfg["episode_len"])

        self.action_space = gym.spaces.Discrete(N_ACTIONS) if gym else range(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        ) if gym else None

        self.seed = seed if seed is not None else int(cfg["seed"])
        self.rng = np.random.default_rng(self.seed)

        self.player_max_hp = float(self.combat["player_hp"])
        self.npc_max_hp = float(self.combat["npc_hp"])
        self.attack_range = float(self.combat["attack_range"])
        self.cone_cos = float(np.cos(np.radians(self.combat["cone"])))
        self.attack_cooldown = float(self.combat["attack_cooldown"])
        self.damage = float(self.combat["damage"])

        self.player = StrategyPlayer(cfg, rng=np.random.default_rng(self.seed + 7))

        self.npc = CombatEntity(self.npc_max_hp, np.zeros(2), np.array([1.0, 0.0]))
        self.npc_pos = self.npc.pos
        self.npc_vel = self.npc.vel
        self.npc_prev_dist = None
        self.step_count = 0
        self.last_info = {}

    # ------------------------------------------------------------------ #
    def reset(self, player_strategy="balanced", npc_pos=None, params_jitter=0.0):
        self.step_count = 0
        pw = self.w
        if npc_pos is None:
            self.npc_pos = self.rng.uniform(pw * 0.2, pw * 0.8, size=2).astype(np.float64)
        else:
            self.npc_pos = np.asarray(npc_pos, dtype=np.float64)
        self.npc.pos = self.npc_pos.copy()
        self.npc.hp = self.npc.max_hp
        self.npc.attack_cooldown = 0.0
        self.npc.facing = np.array([1.0, 0.0], dtype=np.float64)
        self.npc_vel = np.zeros(2, dtype=np.float64)
        self.npc.vel = self.npc_vel

        self.player.reset(player_strategy, params_jitter=params_jitter)
        self.npc_prev_dist = float(np.linalg.norm(self.player.pos - self.npc_pos))
        self.last_info = {"player_strategy": self.player.label, "win": False,
                          "lose": False, "damage_dealt": 0.0, "damage_taken": 0.0}
        return self._obs(), self.last_info

    def set_player_strategy(self, label):
        self.player.set_strategy(label)
    # ------------------------------------------------------------------ #
    def _swing_hits(self, attacker_pos, attacker_facing, target_pos, reach=None):
        """True if a sword swing from attacker would hit target.

        Very-close melee (contact) always hits; otherwise a facing cone applies.
        `reach` optionally overrides the melee range (the interactive demo gives
        the human a slightly longer reach than the NPC's `attack_range`).
        """
        if reach is None:
            reach = self.attack_range
        to_target = target_pos - attacker_pos
        dist = float(np.linalg.norm(to_target))
        if dist > reach or dist < 1e-6:
            return False
        if dist < 0.5:  # scrum: no facing requirement at contact range
            return True
        dirn = to_target / dist
        return float(np.dot(attacker_facing, dirn)) >= self.cone_cos

    def _apply_attack(self, attacker, target):
        """Handle one actor swinging at another. Returns damage dealt."""
        if attacker.attack_cooldown > 0.0:
            return 0.0
        attacker.attack_cooldown = self.attack_cooldown
        reach = self.combat.get("player_reach", None) if attacker is self.player else None
        if self._swing_hits(attacker.pos, attacker.facing, target.pos, reach):
            dmg = self.damage
            target.hp = max(0.0, target.hp - dmg)
            target.attack_cooldown = max(target.attack_cooldown, 1.0)  # hit-stun
            return dmg
        return 0.0

    def step(self, action):
        act = int(action)
        mdir = move_dir_of(act)
        attacking = is_attack(act)
        direction = np.asarray(_DIRS[mdir], dtype=np.float64)

        # ---- NPC movement ----
        desired = direction * self.npc_max_speed
        self.npc_vel = desired
        self.npc.pos = np.clip(self.npc.pos + desired * self.dt, 0.0, self.w)
        sp = float(np.linalg.norm(desired))
        if sp > 1e-4:
            self.npc.facing = desired / sp
        elif attacking:
            # stay to attack: aim at the player
            to_p = self.player.pos - self.npc.pos
            d = float(np.linalg.norm(to_p))
            if d > 1e-6:
                self.npc.facing = to_p / d
        self.npc_pos = self.npc.pos
        self.npc_vel = self.npc.vel

        # cooldown tick before attacks in this step
        self.npc.attack_cooldown = max(0.0, self.npc.attack_cooldown - 1.0)
        self.player.attack_cooldown = max(0.0, self.player.attack_cooldown - 1.0)

        # save pre-step player pos for reward shaping
        pre_step_player = self.player.pos.copy()
        dist_pre = float(np.linalg.norm(pre_step_player - self.npc.pos))

        # ---- player acts ----
        player_pos, pl_attack, pl_facing = self.player.step(self.npc_pos)

        dist = float(np.linalg.norm(player_pos - self.npc.pos))
        delta = dist_pre - dist  # >0 => closed

        # ---- combat resolution ----
        # resolution order: by default the NPC swings first. The interactive
        # demo sets combat["player_first"]=True so an incoming hit never eats
        # a same-frame manual swing (keeps the controls feeling responsive);
        # research configs leave it unset and keep the original ordering.
        dmg_dealt = 0.0
        dmg_taken = 0.0
        if self.combat.get("player_first", False):
            if pl_attack:
                dmg_taken = self._apply_attack(self.player, self.npc)
            if attacking:
                dmg_dealt = self._apply_attack(self.npc, self.player)
        else:
            if attacking:
                dmg_dealt = self._apply_attack(self.npc, self.player)
            if pl_attack:
                dmg_taken = self._apply_attack(self.player, self.npc)

        self.step_count += 1

        # ---- rewards ----
        r_win = float(self.combat["reward_win"])
        r_lose = float(self.combat["reward_lose"])
        r_hit = float(self.combat["reward_hit"])
        r_heading = float(self.combat.get("reward_heading", 0.2))
        r_approach = float(self.combat.get("reward_approach", 1.0))
        r_penalty = float(self.combat.get("reward_penalty", 0.01))

        npc_win = self.player.hp <= 0.0
        npc_lose = self.npc.hp <= 0.0

        reward = 0.0
        reward += r_hit * dmg_dealt
        reward -= r_hit * dmg_taken
        if npc_win:
            reward += r_win
        elif npc_lose:
            reward -= r_lose
        elif not (npc_win or npc_lose):
            # dense shaping only while both alive
            # approach reward: honest "did we get closer" signal
            reward += float(np.clip(delta * r_approach, -0.5, 1.5))
            # heading bonus rewards chasing the player
            dir_to_player = pre_step_player - self.npc.pos
            dn = float(np.linalg.norm(dir_to_player))
            if dn > 1e-6:
                dir_to_player /= dn
            heading_bonus = float(np.dot(dir_to_player, self.npc.facing))
            reward += r_heading * heading_bonus
            # in-range reward: encourages closing to strike distance
            if dist < float(self.combat["attack_range"]):
                reward += float(self.combat.get("reward_inrange", 0.4))
            reward -= r_penalty

        terminated = npc_win or npc_lose
        truncated = self.step_count >= self.episode_len

        self.npc_prev_dist = dist
        self.last_info = {
            "player_strategy": self.player.label,
            "player_pos": player_pos.copy(),
            "npc_pos": self.npc.pos.copy(),
            "dist": dist,
            "win": npc_win,
            "lose": npc_lose,
            "damage_dealt": dmg_dealt,
            "damage_taken": dmg_taken,
            "player_attacked": bool(pl_attack),
            "player_damage_dealt": dmg_taken,
            "player_hp": self.player.hp,
            "npc_hp": self.npc.hp,
            "caught": npc_win,
        }
        return self._obs(), reward, terminated, truncated, self.last_info

    # ------------------------------------------------------------------ #
    def _obs(self):
        d = np.asarray(self.player.pos - self.npc.pos, dtype=np.float64)
        dist = float(np.linalg.norm(d))
        dx, dy = d / self.w
        dist_norm = dist / max(self.sensor_range, 1e-6)

        pv = self.player.vel
        p_speed = float(np.linalg.norm(pv))
        p_heading = pv / p_speed if p_speed > 1e-4 else np.zeros(2)
        n_speed = float(np.linalg.norm(self.npc_vel))
        n_heading = self.npc_vel / n_speed if n_speed > 1e-4 else np.zeros(2)
        closing = (self.npc_prev_dist - dist) if self.npc_prev_dist is not None else 0.0

        ph = self.player.hp / self.player_max_hp
        nh = self.npc.hp / self.npc_max_hp
        pc = self.player.attack_cooldown / max(self.attack_cooldown, 1e-6)
        nc = self.npc.attack_cooldown / max(self.attack_cooldown, 1e-6)

        return np.array(
            [
                dx, dy,
                dist_norm,
                p_heading[0], p_heading[1],
                closing,
                n_heading[0],
                ph, nh,
                pc, nc,
                p_speed / 3.0,
                n_speed / 3.0,
                float(self._swing_hits(self.npc.pos, self.npc.facing, self.player.pos)),
            ],
            dtype=np.float32,
        )

    def obs_dim(self):
        base = 14
        if self.cfg["rl"].get("cond_on_strategy", False):
            return base + len(LABEL_TO_ID)
        return base

    @staticmethod
    def observed_signal(base_obs, strategy_id=None, cond=True):
        base_obs = np.asarray(base_obs, dtype=np.float32)
        if not cond or strategy_id is None:
            return base_obs
        n = len(LABEL_TO_ID)
        hot = np.zeros(n, dtype=np.float32)
        hot[int(strategy_id) % n] = 1.0
        return np.concatenate([base_obs, hot])

    def render(self, *a, **k):
        return None
