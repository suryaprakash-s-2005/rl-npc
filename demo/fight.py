"""Strategy-Adaptive NPC Combat -- polished procedural demo game.

A real-time 1-vs-1 sword fight against the RL NPC trained on strategy-transition-
aware experience sharing. The game world ticks in real time (env dt ~ 0.1 s per
sim step, 60 FPS render) and controls are keyboard-only: WASD to move and a
button to swing, with the sword AUTO-AIMED at the NPC (no mouse needed).

Features
--------
- Sprites are procedural stylized knights (helmet, torso, shield, animated
  legs) -- no external art assets.
- Button attack with AUTO-AIM: SPACE/LMB swings toward the NPC; the hit cone
  is resolved against that direction, so you just need to be in range.
- REAL ADAPTATION SHOWCASE:
    * The NPC runs the strategy-conditioned policy (demo_npc_partitioned.pt,
      cond-on) AND commits to a visible counter game-plan once it locks onto
      your style: KITE (vs aggressive), HUNT (vs evasive), PRESS (vs
      defensive), SWAY (vs balanced). The plan is announced in the arena + HUD.
    * Detection uses a fast soft-lock (~0.5-1 s of play), not just the slow
      CUSUM, so the NPC responds almost immediately when you change style.
    * Style assists 1-4 shape YOUR knight like an AI personality (aggressive /
      defensive / evasive / balanced) so the NPC consistently recognises you
      and you can deliberately trigger -- and feel -- the adaptation.
- Style-test menu (M): pause and pin the autopilot's fighting style so you can
  watch the NPC detect it and adapt.
- You (red) vs the trained strategy-aware NPC (blue).
- Watch mode (T): the player goes autopilot on a real StrategyPlayer personality
  that SWITCHES mid-fight (aggressive -> defensive -> evasive -> balanced at
  step 100), letting you watch the NPC detect and adapt -- the paper showcase.
  You can also force a switch with 1-4 at any time.

Controls
-------
  WASD / arrows        : move
  SPACE / LMB (hold)   : swing your sword -- AUTO-AIMED at the NPC
  M                    : style-test menu (pause) -- pick which fighting
                         style the player uses so the NPC has to adapt
  T                    : toggle watch mode (autopilot plays for you)
  1 / 2 / 3 / 4        : style assist -- you fight like aggressive /
                         defensive / evasive / balanced (press again to clear)
                         in WATCH mode 1-4 switches the autopilot personality
  R                    : reset episode
  Q / ESC              : quit

Usage:
    python -m demo.fight [model_path] [--frames N] [--shot out.png]
        model_path defaults to data/demo_npc_partitioned.pt
        --frames N auto-quits after N rendered frames (for headless checks)
        --shot  out.png  saves the first rendered frame to a PNG and quits
"""

import os
import sys
import time
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
import pygame

from src.config import load_config
from src.env.chase_env import ChaseEnv, N_ACTIONS, is_attack, _DIRS
from src.env.features import FeatureWindow
from src.rl.dqn_agent import DQNAgent
from src.strategy.detector import FeatureHistory, StrategyClassifier
from src.strategy.transitions import CusumTransitionDetector

LABELS = ["aggressive", "defensive", "evasive", "balanced"]
DIR_NAMES = ["stay", "E", "W", "S", "N", "SE", "NE", "SW", "NW"]
N = len(LABELS)

PERSONA_COLOR = {
    "aggressive": (255, 90, 90),
    "defensive": (110, 235, 130),
    "evasive": (190, 130, 255),
    "balanced": (255, 205, 90),
}

# style-test menu: pick which fighting style the PLAYER uses and challenge the
# NPC to adapt. None = the rotating autopilot showcase (paper demo).
MENU_ITEMS = [
    ("AUTO-CYCLE", None,        "rotating personalities (paper showcase)"),
    ("AGGRESSIVE", 0,           "you rush -- NPC should KITE"),
    ("DEFENSIVE", 1,            "you back off -- NPC should PRESS"),
    ("EVASIVE", 2,              "you strafe -- NPC should HUNT"),
    ("BALANCED", 3,             "you mix -- NPC should SWAY"),
]

WORLD_PX = 640
HUD_H = 76
WIN_W = WORLD_PX
WIN_H = WORLD_PX + HUD_H
FPS = 60
SIM_DT = 0.1          # one env step per SIM_DT seconds of real time (1x speed)
SWITCH_STEP = 100     # sim steps after which watch mode switches personality

# Fun-tuned combat (in-game override; research config stays untouched)
FUN_COMBAT = {
    "player_hp": 60,
    "npc_hp": 60,
    "damage": 3,
    "attack_cooldown": 6,
    "attack_range": 1.0,
    "player_reach": 1.15,   # human swing connects near the visible blade tip
    "player_first": True,   # player swing resolves before NPC (responsive)
}

# NPC counter game-plans keyed on the detected player style. These are
# interpretable "intents" the NPC commits to once it locks onto your style
# (blended with the trained RL policy by detection confidence), so the
# adaptation is plainly observable: you switch style -> the NPC visibly
# switches game plan.
PLAN_OF_STYLE = {
    "aggressive": "KITE",     # you rush:  the NPC backs off and slices at range
    "defensive": "PRESS",     # you back off: it closes and presses the attack
    "evasive": "HUNT",        # you run:    it pursues relentlessly
    "balanced": "SWAY",       # you're mixed: it relies on its learned policy
}
PLAN_EXPLAIN = {
    "KITE": "evades you, strikes at range",
    "HUNT": "chases you down",
    "PRESS": "closes in and pressures you",
    "SWAY": "adapting as it goes",
}
PLAN_COLOR = {
    "KITE": (255, 186, 96),
    "HUNT": (120, 205, 255),
    "PRESS": (255, 120, 110),
    "SWAY": (205, 215, 230),
}

# human style assists (keys 1-4): steer your knight like the corresponding
# AI personality so the online detector locks on you quickly, letting you
# EXPERIENCE the NPC adapting when you change style mid-fight.
ASSIST_KEYS = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4}
ASSIST_NAME = {1: "AGGR", 2: "DEF", 3: "EVA", 4: "BAL"}

DEFAULT_MODEL = os.path.join("data", "demo_npc_partitioned.pt")
FALLBACK_MODEL = os.path.join("data", "npc_partitioned-gt-seed0.pt")


# --------------------------------------------------------------------------- #
# small math helpers
# --------------------------------------------------------------------------- #
def _v2(a):
    return np.asarray(a, dtype=np.float64)


def _rot(v, ang):
    c, s = np.cos(ang), np.sin(ang)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def _norm(v):
    d = float(np.linalg.norm(v))
    return v / d if d > 1e-9 else np.zeros(2)


# --------------------------------------------------------------------------- #
# arena backdrop (precomputed once)
# --------------------------------------------------------------------------- #
def build_arena(size_px):
    surf = pygame.Surface((size_px, size_px))
    rng = np.random.default_rng(7)
    tile = 40
    base = (26, 30, 42)
    alt = (30, 35, 49)
    for j in range(0, size_px, tile):
        for i in range(0, size_px, tile):
            c = base if ((i // tile) + (j // tile)) % 2 == 0 else alt
            pygame.draw.rect(surf, c, (i, j, tile, tile))
    # mottled floor
    for _ in range(900):
        x = int(rng.integers(0, size_px))
        y = int(rng.integers(0, size_px))
        m = int(rng.integers(26, 44))
        pygame.draw.circle(surf, (m, m + 4, m + 8), (x, y), int(rng.integers(1, 3)))
    # warm center light pool
    glow = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    for r in range(size_px // 2, 0, -20):
        a = int(26 * (1 - r / (size_px / 2)))
        pygame.draw.circle(glow, (255, 220, 150, max(0, a)),
                           (size_px // 2, size_px // 2), r)
    surf.blit(glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    # vignette
    vig = pygame.Surface((size_px, size_px), pygame.SRCALPHA)
    for r in range(size_px // 2, 0, -12):
        a = int(160 * (r / (size_px / 2)))
        pygame.draw.circle(vig, (0, 0, 10, a), (size_px // 2, size_px // 2), r)
    surf.blit(vig, (0, 0))
    # arena border with glow
    pygame.draw.rect(surf, (90, 105, 140), (0, 0, size_px, size_px), 3)
    for g in (12, 8, 4):
        pygame.draw.rect(surf, (70, 85, 125, 18),
                         (g, g, size_px - 2 * g, size_px - 2 * g), 3)
    return surf


def build_glow_sprite(rad, color):
    s = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
    for r in range(rad, 0, -2):
        k = 1 - r / rad
        a = int(70 * (1 - k))
        pygame.draw.circle(s, (*color[:3], a), (rad + 2, rad + 2), r)
    return s


# --------------------------------------------------------------------------- #
# effect / particle helpers
# --------------------------------------------------------------------------- #
class Fx:
    def __init__(self):
        self.parts = []      # [x, y, vx, vy, life, ttl, color, size, grav]
        self.nums = []       # [x, y, text, color, ttl]
        self.shake = 0.0
        self.flash = 0.0     # 0..1 white overlay

    def add_sparks(self, pos, color, n=14, speed=160.0):
        rng = np.random.default_rng()
        for _ in range(n):
            a = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(speed * 0.35, speed)
            self.parts.append([pos[0], pos[1],
                               np.cos(a) * sp, np.sin(a) * sp,
                               0.0, rng.uniform(0.25, 0.6),
                               color, int(rng.uniform(1, 4)), 300.0])

    def add_burst(self, pos, color, n=36, speed=260.0):
        rng = np.random.default_rng()
        for _ in range(n):
            a = rng.uniform(0, 2 * np.pi)
            sp = rng.uniform(speed * 0.2, speed)
            self.parts.append([pos[0], pos[1],
                               np.cos(a) * sp, np.sin(a) * sp,
                               0.0, rng.uniform(0.3, 0.9),
                               color, int(rng.uniform(2, 5)), 220.0])

    def add_num(self, pos, text, color):
        self.nums.append([pos[0], pos[1], text, color, 0.9])

    def shake_with(self, mag):
        self.shake = min(12.0, max(self.shake, mag))

    def flash_with(self, v):
        self.flash = min(1.0, max(self.flash, v))

    def update(self, dt):
        for p in self.parts:
            p[4] += dt
            p[1] += p[3] * dt
            p[2] *= (1.0 - max(0.0, (dt * 2.4)))
            p[3] += p[8] * dt
        self.parts = [p for p in self.parts if p[4] < p[5]]
        for n in self.nums:
            n[1] -= 28 * dt
            n[4] -= dt
        self.nums = [n for n in self.nums if n[4] > 0]
        self.shake = max(0.0, self.shake - 40 * dt)
        self.flash = max(0.0, self.flash - 2.2 * dt)

    def draw(self, screen):
        rng = np.random.default_rng()
        for x, y, vx, vy, t, ttl, color, size, _g in self.parts:
            k = 1 - t / ttl
            xx, yy = x + (rng.normal() * 2), y + (rng.normal() * 2)
            pygame.draw.line(screen, color, (xx, yy),
                             (xx - vx * 0.03, yy - vy * 0.03), max(1, size))
        for x, y, text, color, ttl in self.nums:
            a = max(0, min(255, int(255 * min(1.0, ttl))))
            small = pygame.font.SysFont("consolas", 16).render(
                text, True, (*color, a))
            screen.blit(small, (x - small.get_width() // 2, y))


# --------------------------------------------------------------------------- #
# procedural sound -- synthesized at startup, no audio assets
# --------------------------------------------------------------------------- #
def _sfx_sound(dur, f0, f1, kind, vol):
    sr = 22050
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    if kind == "noise":
        w = np.random.default_rng().normal(0.0, 1.0, n)
        w = np.convolve(w, np.ones(24) / 24, mode="same")
    else:
        ph = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * max(dur, 1e-9)))
        w = np.sin(ph)
    w = w * np.exp(-3.5 * t / max(dur, 1e-9))
    w = np.clip(w * vol * 0.5, -1.0, 1.0)
    st = np.stack([w, w], axis=1)
    return pygame.sndarray.make_sound(np.int16(st * 32767))


class Sfx:
    SFX = {
        "whoosh_p": (0.16, 260, 780, "sweep", 0.45),
        "whoosh_n": (0.16, 340, 940, "sweep", 0.45),
        "clink_p":  (0.07, 980, 540, "sweep", 0.5),   # you landed a hit
        "clink_n":  (0.07, 640, 380, "sweep", 0.6),   # you got hit
        "boom_n":   (0.5, 300, 60, "sweep", 0.75),    # NPC defeated
        "boom_p":   (0.5, 260, 50, "sweep", 0.7),     # you died
        "plan":     (0.2, 640, 330, "sweep", 0.5),    # NPC locks a game-plan
        "lock":     (0.12, 880, 1180, "sweep", 0.35), # detection lock-on
    }

    def __init__(self):
        self.ok = False
        self.sounds = {}
        try:
            pygame.mixer.quit()
            pygame.mixer.pre_init(22050, -16, 2, 512)
            pygame.mixer.init()
        except Exception:
            return
        try:
            if pygame.mixer.get_init() is not None:
                pygame.mixer.set_num_channels(8)
                for name, sp in self.SFX.items():
                    self.sounds[name] = _sfx_sound(*sp)
                self.ok = True
        except Exception:
            self.ok = False

    def play(self, name):
        if self.ok and name in self.sounds:
            try:
                self.sounds[name].play()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# sword swing animation
# --------------------------------------------------------------------------- #
class Sword:
    """State machine for a sword swing: windup -> swing -> recover -> idle."""

    DUR_WINDUP = 0.10
    DUR_SWING = 0.20
    DUR_RECOVER = 0.18
    SWEEP = 2.5  # radians swept across

    def __init__(self):
        self.phase = "idle"
        self.t = 0.0
        self.ang = 0.0   # current blade angle offset (radians)
        self.sweep_dir = 1.0

    def update(self, dt):
        if self.phase == "idle":
            self.ang = 0.0
            return
        self.t += dt
        if self.phase == "windup":
            k = min(1.0, self.t / self.DUR_WINDUP)
            self.ang = -self.sweep_dir * 1.9 * k
            if k >= 1.0:
                self.phase, self.t = "swing", 0.0
        elif self.phase == "swing":
            k = min(1.0, self.t / self.DUR_SWING)
            self.ang = self.sweep_dir * (-1.4 + (self.SWEEP + 0.6) * k)
            if k >= 1.0:
                self.phase, self.t = "recover", 0.0
        elif self.phase == "recover":
            k = min(1.0, self.t / self.DUR_RECOVER)
            self.ang = self.sweep_dir * 1.2 * (1.0 - k)
            if k >= 1.0:
                self.phase, self.t = "idle", 0.0

    def trigger(self):
        if self.phase == "idle":
            self.phase = "windup"
            self.t = 0.0
            self.sweep_dir = 1.0 if np.random.random() < 0.5 else -1.0

    @property
    def trail_active(self):
        return self.phase == "swing"


def blade_poly(pivot, facing, ang, scale=1.0):
    """World-space sword geometry. `pivot` is the hand (px), the blade points
    along `facing` rotated by `ang` about the pivot. Returns named points so the
    drawer can layer outline / shading / hilt separately.

    Layout (all distances in knight-radius units along `bd`, widths along `per`):
        tip           1.9  (matches attack_range=1.0 from a 0.55r hand pivot,
                            i.e. (1.0*32 - 0.55*13) / 13 px scaled to r-units;
                            a compact blade so knights stay clearly visible)
        edge run      1.40  parallel straight-edge blade (width 0.27)
        blade base    0.35  half-width 0.27
        guard         0.22..0.34  half-width 0.28
        grip          0.07..0.21  half-width 0.08
        pommel        0.03
    """
    c, s = np.cos(ang), np.sin(ang)
    fx, fy = facing
    bd = np.array([fx * c - fy * s, fx * s + fy * c], dtype=np.float64)
    px, py = -bd[1], bd[0]
    bd = bd * scale
    per = np.array([px, py], dtype=np.float64) * scale
    tip = pivot + bd * 1.9
    e_top = pivot + bd * 1.40 + per * (0.27 * scale)
    e_bot = pivot + bd * 1.40 - per * (0.27 * scale)
    b_top = pivot + bd * 0.35 + per * (0.27 * scale)
    b_bot = pivot + bd * 0.35 - per * (0.27 * scale)
    base_mid = pivot + bd * 0.35
    guard = [
        pivot + bd * 0.22 + per * (0.28 * scale),
        pivot + bd * 0.34 + per * (0.28 * scale),
        pivot + bd * 0.34 - per * (0.28 * scale),
        pivot + bd * 0.22 - per * (0.28 * scale),
    ]
    grip = [
        pivot + bd * 0.21 + per * (0.08 * scale),
        pivot + bd * 0.07 + per * (0.08 * scale),
        pivot + bd * 0.07 - per * (0.08 * scale),
        pivot + bd * 0.21 - per * (0.08 * scale),
    ]
    pommel = pivot + bd * 0.03
    return {
        "tip": tip, "b_top": b_top, "b_bot": b_bot, "base_mid": base_mid,
        "e_top": e_top, "e_bot": e_bot,
        "guard": guard, "grip": grip, "pommel": pommel,
    }


# --------------------------------------------------------------------------- #
# fighter rendering -- stylized knight (procedural, no external assets)
# --------------------------------------------------------------------------- #
KNIGHT_PALETTES = {
    "player": {
        "tunic": (178, 52, 50), "metal": (170, 175, 192),
        "trim": (216, 168, 62), "plume": (255, 96, 78),
        "shield": (166, 44, 44), "shield_rim": (222, 176, 70),
        "boot": (92, 50, 48), "leg": (52, 46, 58),
    },
    "npc": {
        "tunic": (54, 102, 184), "metal": (176, 182, 200),
        "trim": (196, 202, 218), "plume": (96, 160, 255),
        "shield": (48, 94, 182), "shield_rim": (204, 210, 226),
        "boot": (50, 72, 114), "leg": (46, 50, 64),
    },
}


def _lighten(c, k):
    return tuple(min(255, int(x * k)) for x in c)


class Knight:
    def __init__(self, palette, color, glow_color, r=13):
        self.pal = palette
        self.color = color
        self.glow_color = glow_color
        self.r = r
        self.sword = Sword()
        self.flash_t = 0.0
        self.glow = build_glow_sprite(r * 2, glow_color)
        self.t = 0.0
        self.walk = 0.0
        self.leg_phase = 0.0
        # --- pre-rendered parts ------------------------------------------- #
        bs = int(r * 3.2)
        self.body_sz = bs
        body = pygame.Surface((bs, bs), pygame.SRCALPHA)
        pygame.draw.ellipse(body, self.pal["metal"], (0, 0, bs, bs))
        pygame.draw.ellipse(body, self.pal["tunic"],
                            (bs * 0.18, bs * 0.14, bs * 0.64, bs * 0.72))
        bl = max(2, int(r * 0.26))
        pygame.draw.line(body, self.pal["trim"],
                         (int(bs * 0.14), int(bs * 0.62)),
                         (int(bs * 0.86), int(bs * 0.62)), bl)
        pygame.draw.ellipse(body, self.pal["trim"],
                            (bs * 0.40, bs * 0.06, bs * 0.20, bs * 0.16))
        self.body = body
        # helmet (visor slit faces +x in the pre-render)
        hs = int(r * 3.1)
        helm = pygame.Surface((hs, hs), pygame.SRCALPHA)
        pygame.draw.ellipse(helm, self.pal["metal"],
                            (hs * 0.18, hs * 0.26, hs * 0.64, hs * 0.60))
        pygame.draw.polygon(helm, self.pal["metal"],
                            [(hs * 0.18, int(hs * 0.62)), (int(hs * 0.82), int(hs * 0.62)),
                             (int(hs * 0.70), int(hs * 0.92)), (int(hs * 0.30), int(hs * 0.92))])
        # visor slit
        pygame.draw.rect(helm, (22, 24, 32),
                         (int(hs * 0.10), int(hs * 0.48), int(hs * 0.80), int(hs * 0.07)))
        pygame.draw.rect(helm, (150, 158, 178),
                         (int(hs * 0.08), int(hs * 0.47), int(hs * 0.82), 2))
        # plume on top
        pygame.draw.arc(helm, self.pal["plume"],
                        (int(hs * 0.14), int(hs * 0.10), int(hs * 0.72), int(hs * 0.34)),
                        3.3, 6.1, max(2, int(r * 0.42)))
        self.helm = helm
        # round shield (face-on)
        ss = int(r * 2.3)
        sh = pygame.Surface((ss, ss), pygame.SRCALPHA)
        pygame.draw.circle(sh, self.pal["shield_rim"], (ss // 2, ss // 2), ss // 2)
        pygame.draw.circle(sh, self.pal["shield"], (ss // 2, ss // 2), ss // 2 - 3)
        pygame.draw.circle(sh, _lighten(self.pal["shield"], 1.35),
                           (ss // 2, ss // 2), int(ss * 0.14))
        pygame.draw.circle(sh, self.pal["shield_rim"],
                           (ss // 2, ss // 2), int(ss * 0.26), 2)
        self.shield_sz = ss
        self.shield = sh
        self.persona_ring_c = None

    def update(self, dt, world_pos, facing, speed=0.0):
        self.sword.update(dt)
        self.t += dt
        self.flash_t = max(0.0, self.flash_t - dt)
        if speed > 0.05:
            self.leg_phase += dt * speed * 9.0
        return world_pos

    def flash(self, mag=0.5):
        self.flash_t = mag

    # ------------------------------------------------------------------ #
    def draw(self, screen, scale, world_pos, facing, persona_c=None):
        cx, cy = int(world_pos[0] * scale), int(world_pos[1] * scale)
        r = self.r
        f = _v2(facing); fn = float(np.linalg.norm(f))
        if fn < 1e-6:
            f = np.array([1.0, 0.0])
        else:
            f = f / fn
        p = np.array([-f[1], f[0]])
        bob = np.sin(self.t * 2.4) * 0.7

        # shadow
        pygame.draw.ellipse(screen, (10, 12, 18, 110),
                            (cx - r, cy + 3, r * 2, r + 3))
        screen.blit(self.glow, (cx - r * 2 - 2, cy - r * 2 - 2),
                    special_flags=pygame.BLEND_RGBA_ADD)

        hip = np.array([cx, cy]) + f * (0.0) - np.array([0.0, r * 0.15])
        for sgn in (1.0, -1.0):
            side = p * (r * 0.30) * sgn
            opp = -sgn
            foot = np.array([cx, cy]) + f * (np.sin(self.leg_phase + sgn * 1.1) * r * 0.50) \
                + side * 0.9 + np.array([0.0, r * 0.82])
            pygame.draw.line(screen, self.pal["leg"],
                             (int(hip[0] + side[0]), int(hip[1] + side[1] + bob)),
                             (int(foot[0]), int(foot[1])),
                             max(2, int(r * 0.26)))
            pygame.draw.circle(screen, self.pal["boot"],
                               (int(foot[0]), int(foot[1])), max(2, int(r * 0.24)))

        # torso
        screen.blit(self.body, (cx - self.body_sz // 2, int(cy - self.body_sz // 2 + bob)))
        # shoulder pauldron on the facing side
        sh_x, sh_y = int(cx + f[0] * r * 0.52), int(cy - r * 0.28 + bob)
        pygame.draw.circle(screen, self.pal["metal"], (sh_x, sh_y), int(r * 0.34))
        pygame.draw.circle(screen, self.pal["trim"], (sh_x, sh_y), int(r * 0.30), 2)

        # shield, face-on, held on the lead side
        ssz = self.shield_sz
        scx = int(cx + f[0] * r * 0.32) - ssz // 2
        scy = int(cy - r * 0.42 + bob) - ssz // 2
        screen.blit(self.shield, (scx, scy))

        # helmet, rotated so the visor faces the motion direction
        hsz = self.helm.get_width()
        ang = math.degrees(math.atan2(f[1], f[0]))
        rot = pygame.transform.rotate(self.helm, -ang)
        hx = int(cx + f[0] * r * 0.30) - rot.get_width() // 2
        hy = int(cy - r * 0.98) + bob - rot.get_height() // 2
        screen.blit(rot, (hx, hy))

        # sword from the hand pivot
        self._draw_sword(screen, cx, cy, f, p, bob)

        if self.flash_t > 0:
            ring = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring, (255, 255, 255, int(255 * self.flash_t)),
                               (r + 2, r + 2), r + 1)
            screen.blit(ring, (cx - r - 2, int(cy - r - 2 + bob)))

        # persona ring
        if persona_c is not None:
            pygame.draw.circle(screen, persona_c,
                               (cx, int(cy + bob)), r + 5, 2)

    def _draw_sword(self, screen, cx, cy, f, p, bob):
        r = self.r
        pivot = np.array([cx + f[0] * r * 0.55 + p[0] * r * 0.30,
                          cy + f[1] * r * 0.55 + p[1] * r * 0.30 + bob])
        ang = self.sword.ang
        scl = float(r)
        g = blade_poly(pivot, f, ang, scale=scl)
        px = lambda v: (int(v[0]), int(v[1]))

        # slash trail while swinging (filled wedge under the blade)
        if self.sword.trail_active:
            k = min(1.0, self.sword.t / self.sword.DUR_SWING)
            base_ang = np.arctan2(f[1], f[0])
            a0 = base_ang + self.sword.sweep_dir * (-1.6)
            a1 = base_ang + self.sword.sweep_dir * (-1.6 + 1.9 * k)
            R = float(np.linalg.norm(g["tip"] - pivot))
            trail = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            rect = pygame.Rect(int(pivot[0] - R), int(pivot[1] - R),
                               int(2 * R), int(2 * R))
            pygame.draw.arc(trail, (235, 245, 255, int(70 * (1 - k * 0.5))),
                            rect, a0, a1, 16)
            pygame.draw.arc(trail, (205, 225, 245, int(105 * (1 - k * 0.4))),
                            rect, a0, a1, 9)
            pygame.draw.arc(trail, (248, 250, 255, int(150 * (1 - k * 0.5))),
                            rect, a0, a1, 4)
            screen.blit(trail, (0, 0))

        # silhouette (dark outline under everything)
        sil = blade_poly(pivot, f, ang, scale=scl * 1.06)
        sil_pts = [sil["tip"], sil["e_top"], sil["b_top"], sil["guard"][1],
                   sil["guard"][0], sil["grip"][1], sil["pommel"], sil["grip"][2],
                   sil["guard"][3], sil["guard"][2], sil["b_bot"], sil["e_bot"]]
        pygame.draw.polygon(screen, (38, 44, 58), [px(q) for q in sil_pts])

        # blade: straight edges, fuller, point -- shaded steel
        pygame.draw.polygon(screen, (148, 160, 184),
                            [px(g["tip"]), px(g["e_bot"]), px(g["b_bot"]),
                             px(g["base_mid"])])
        pygame.draw.polygon(screen, (216, 228, 244),
                            [px(g["tip"]), px(g["base_mid"]), px(g["b_top"]),
                             px(g["e_top"])])
        # fuller (darker groove down the blade spine)
        pygame.draw.line(screen, (205, 218, 236), px(g["base_mid"]),
                         px(g["tip"]), 1)
        # bright edge sheen on the true edge + point highlight
        pygame.draw.line(screen, (250, 253, 255), px(g["e_top"]), px(g["tip"]), 1)
        pygame.draw.circle(screen, (255, 255, 255), px(g["tip"]), 1)

        # crossguard (brass) with darker edge
        grd = [px(q) for q in g["guard"]]
        pygame.draw.polygon(screen, (196, 148, 58), grd)
        pygame.draw.polygon(screen, (120, 88, 44), grd, 2)
        pygame.draw.line(screen, (255, 224, 150), grd[0], grd[3], 2)

        # grip (brown leather wrap) + pommel cap
        grp = [px(q) for q in g["grip"]]
        pygame.draw.polygon(screen, (92, 70, 58), grp)
        pygame.draw.line(screen, (58, 44, 40), grp[0], grp[3], 2)
        pygame.draw.circle(screen, (150, 158, 178), px(g["pommel"]), 3)
        pygame.draw.circle(screen, (40, 44, 56), px(g["pommel"]), 3, 1)

        # gauntlet (hand) over the grip: small dark fist + finger guard
        pygame.draw.circle(screen, (116, 122, 140), (int(pivot[0]), int(pivot[1])), 6)
        pygame.draw.circle(screen, (40, 44, 56), (int(pivot[0]), int(pivot[1])), 6, 1)


# --------------------------------------------------------------------------- #
# game
# --------------------------------------------------------------------------- #
class Game:
    def __init__(self, cfg, model_path, frames=None, shot_at=None):
        self.cfg = cfg
        self.frames = frames
        self.frame_n = 0
        self.shot_at = shot_at if shot_at else 1
        self.running = True

        # detector
        self.det_model = StrategyClassifier(cfg)
        import torch
        ck = torch.load(cfg["detector"]["model_path"], map_location="cpu")
        self.det_model.load_state_dict(ck["state_dict"])
        self.det_model.eval()

        # auto-detect cond flag from the NPC model
        ck = torch.load(model_path, map_location="cpu")
        self.in_dim = int(ck["online"]["net.0.weight"].shape[1])
        cfg["rl"]["cond_on_strategy"] = self.in_dim > 14

        # fun combat override (in-memory only)
        for k, v in FUN_COMBAT.items():
            cfg["combat"][k] = v
        cfg["env"]["npc_max_speed"] = 2.5   # demo pace: close to player speed so
                                            # the standoff never gets corner-squeezed

        self.model_path = model_path
        self.env = ChaseEnv(cfg, seed=int(cfg["seed"]))
        self.agent = DQNAgent(cfg, self.env.obs_dim(), N_ACTIONS)
        self.agent.load(model_path)

        self.scale = WORLD_PX / self.env.w
        self.arena = build_arena(WORLD_PX)
        self.fx = Fx()
        self.player_v = Knight(KNIGHT_PALETTES["player"], (235, 90, 80), (190, 55, 45))
        self.npc_v = Knight(KNIGHT_PALETTES["npc"], (80, 150, 230), (45, 105, 195))

        pygame.init()
        pygame.display.set_caption("Strategy-Adaptive NPC -- real-time combat")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        try:
            pygame.mouse.set_visible(False)
        except Exception:
            pass
        self.hud_font = pygame.font.SysFont("consolas", 16)
        self.hud_font_s = pygame.font.SysFont("consolas", 14)
        self.big_font = pygame.font.SysFont("consolas", 30, bold=True)
        self.clock = pygame.time.Clock()

        self.watch = False
        self.kills = 0
        self.deaths = 0
        self.transitions = 0
        self.adapt_events = 0
        self.detected = None        # soft-locked style currently counter-adapted to
        self.declared = None        # VS-mode declared style (sticky while key held)
        self.soft_conf = 0.6        # confidence of the last soft-lock
        self.soft_streak = 0
        self.npc_plan = "SWAY"
        self.plan_t = 0.0
        self.plan_flash = 0.0
        self.banner_text = None     # big center-screen adaptation announcement
        self.banner_color = (255, 255, 255)
        self.banner_t = 0.0
        self.style_assist = 0       # 1..4 style assist, 0 = none
        self.session_label = "balanced"
        self.menu_open = False   # style-test menu overlay (pauses sim)
        self.menu_sel = 0        # highlighted MENU_ITEMS index
        self.fixed_style = None  # None = auto-cycle; else LABELS index fixed
        self.reward = 0.0
        self.over = False
        self.result = None
        self.episode_idx = 0
        self.over_t = 0.0
        self.assist_phase = 0.0
        self.sfx = Sfx()

        self.sim_acc = 0.0
        self.shot = None
        self.prev_npc = self.env.npc_pos.copy()
        self.prev_player = self.env.player.pos.copy()
        self.prev_pcd = 0.0
        self.prev_ncd = 0.0
        self.fw = FeatureWindow(cfg["strategy"]["feature_window"])
        self.fh = FeatureHistory(cfg["detector"]["context_len"])
        self.cusum = CusumTransitionDetector(cfg, N)

        self.last = time.perf_counter()
        self.reset_episode(watch=False)

    # ------------------------------------------------------------------ #
    def reset_episode(self, watch=None):
        if watch is not None:
            self.watch = watch
            if self.watch:
                self.declared = None
        label = (LABELS[self.fixed_style] if self.fixed_style is not None
         else LABELS[self.episode_idx % N])
        self.episode_idx += 1
        self.env.reset(player_strategy=label)
        self.env.player.clear_manual() if self.watch else None
        if not self.watch:
            self.env.player.set_manual_velocity([0.0, 0.0])
        self.fw = FeatureWindow(self.cfg["strategy"]["feature_window"])
        self.fh = FeatureHistory(self.cfg["detector"]["context_len"])
        self.cusum = CusumTransitionDetector(self.cfg, N)
        self.prev_npc = self.env.npc_pos.copy()
        self.prev_player = self.env.player.pos.copy()
        self.prev_pcd = 0.0
        self.prev_ncd = 0.0
        self.detected = None
        self.soft_conf = 0.6
        self.soft_streak = 0
        self.adapt_events = 0
        self.det_history = collections.deque(maxlen=16)
        if self.declared is not None:
            self.detected = self.declared
            self.npc_plan = PLAN_OF_STYLE[LABELS[self.declared]]
            self.det_history.append(self.declared)
        else:
            self.npc_plan = "SWAY"
        self.plan_t = 0.0
        self.plan_flash = 0.0
        self.banner_text = None
        self.banner_sub = ""
        self.banner_t = 0.0
        self.style_assist = 0
        self.assist_phase = 0.0
        self.attack_held = False
        self.session_label = label
        self.reward = 0.0
        self.over = False
        self.result = None
        self.transitions = 0
        self.sim_acc = 0.0
        self.fx = Fx()
        self.player_v = Knight(KNIGHT_PALETTES["player"], (235, 90, 80), (190, 55, 45))
        self.npc_v = Knight(KNIGHT_PALETTES["npc"], (80, 150, 230), (45, 105, 195))

    # ------------------------------------------------------------------ #
    def handle_input(self, keys, attack_key):
        if not self.watch:
            ux = (int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) -
                  int(keys[pygame.K_LEFT] or keys[pygame.K_a]))
            uy = (int(keys[pygame.K_DOWN] or keys[pygame.K_s]) -
                  int(keys[pygame.K_UP] or keys[pygame.K_w]))
            # one swing per press: a fresh attack only on the rising edge, so
            # holding SPACE does not continuously drain enemy HP
            down = bool(attack_key)
            attack_edge = down and not self.attack_held
            self.attack_held = down
            attack = attack_edge
            speed = self.env.env_cfg["npc_max_speed"] * 0.9
            if ux or uy:
                mv = np.array([ux, uy], dtype=np.float64)
                mv = mv / max(np.linalg.norm(mv), 1e-6) * speed
            else:
                mv = np.zeros(2)
            mv = self.assist_vel(mv, speed)
            attack = attack or self.assist_attack()
            aim = None
            if attack:
                # button attack with AUTO-AIM: face the NPC, no mouse needed
                to_n = self.env.npc.pos - self.env.player.pos
                dn = float(np.linalg.norm(to_n))
                if dn > 1e-6:
                    aim = to_n / dn
            if float(np.linalg.norm(mv)) > 1e-4 or attack:
                self.env.player.set_manual_velocity(mv, attack=attack, aim=aim)
            else:
                self.env.player.set_manual_velocity(np.zeros(2))

    # ------------------------------------------------------------------ #
    # style assists (keys 1-4) blend YOUR steering with the AI persona's
    # own steering (`StrategyPlayer._steer`), so your knight behaves like
    # the corresponding training personality and the online detector locks
    # on quickly and reliably -- letting you trigger and FEEL the NPC adapt.
    def assist_vel(self, manual, speed):
        s = self.style_assist
        env = self.env
        if s == 0:
            if env.player.label != self.session_label:
                env.player.set_strategy(self.session_label)
            return manual
        env.player.set_strategy(LABELS[s - 1])
        steer = env.player._steer(env.npc.pos)
        v = manual * 0.6 + steer * 0.85
        vn = float(np.linalg.norm(v))
        if vn > speed:
            v = v / vn * speed
        return v

    def assist_attack(self):
        if self.style_assist == 0:
            return False
        return bool(self.env.player._attack_decision(self.env.npc.pos))

    # ------------------------------------------------------------------ #
    # NPC counter game-plan: pick an interpretable intent action from the
    # detected style, blended with the RL action by detection confidence.
    def plan_action(self, plan):
        env = self.env
        to_np = env.player.pos - env.npc.pos
        d = float(np.linalg.norm(to_np))
        t = to_np / d if d > 1e-4 else np.zeros(2)
        vel = np.zeros(2)
        attack = False
        cd_ok = env.npc.attack_cooldown <= 0.0
        if plan == "HUNT":
            vel = t
            attack = cd_ok and d < env.attack_range
        elif plan == "KITE":
            if d > 1e-4 and d < 5.0:
                vel = -t
                attack = cd_ok and 0.7 < d < env.attack_range + 0.6
            elif d >= 5.0:
                vel = t * 0.5
        elif plan == "PRESS":
            if d > 1.5:
                vel = t
            elif d < 1.0:
                vel = -t
            attack = cd_ok and d < env.attack_range + 0.2
        vn = float(np.linalg.norm(vel))
        if vn < 1e-3:
            mdir = 0
        else:
            best = max(range(1, 9), key=lambda i: float(np.dot(vel, _DIRS[i])))
            mdir = best
        return mdir if not attack else mdir + 9

    def run_step(self):
        """Advance one sim step; returns effects info."""
        env = self.env
        # NPC policy action -- conditioned on the (soft-locked) detected
        # style so the RL state itself changes with your play.
        cond = bool(self.cfg["rl"]["cond_on_strategy"])
        label_id = self.detected if self.detected is not None else LABELS.index("balanced")
        obs = env.observed_signal(env._obs(), label_id, cond)
        action = self.agent.greedy(obs)

        # --- NPC counter game-plan: blend rule intent with the RL action --- #
        plan = PLAN_OF_STYLE[LABELS[label_id]]
        if plan != "SWAY":
            rule_a = self.plan_action(plan)
            p_rule = min(0.8, 0.28 + 0.42 * min(1.0, self.soft_conf))
            if np.random.random() < p_rule:
                action = rule_a

        # --- standoff governor: never merge with the player; hold at blade
        #     range and strike from there (attack_range = 1.0, bodies ~0.8
        #     units wide, blade tip ~1.0) ------------------------------------ #
        d = float(np.linalg.norm(env.player.pos - env.npc.pos))
        away = env.npc.pos - env.player.pos
        an = float(np.linalg.norm(away))
        mdir = int(action) % 9
        if d < 1.15 and an > 1e-5:
            away /= an
            if d < 0.82:
                # true scrum (sprites overlapping): back out, but never into
                # a wall (that would pin us in the corner and look buggy).
                # Prefer the legal retreat that maximizes distance; if the
                # retreat axis is blocked, slide along the wall instead.
                pad = 0.12
                reach = env.npc_max_speed * env.dt * 1.5
                legal = [i for i in range(1, 9)
                         if (pad <= env.npc.pos[0] + _DIRS[i][0] * reach <= env.w - pad
                             and pad <= env.npc.pos[1] + _DIRS[i][1] * reach <= env.w - pad)]
                if legal:
                    action = max(legal, key=lambda i: float(np.dot(away, _DIRS[i])))
                elif (env.npc.pos[0] > pad + reach
                      and env.npc.pos[0] < env.w - pad - reach):
                    action = 1 if _DIRS[1][0] * away[0] >= 0 else 2
                elif (env.npc.pos[1] > pad + reach
                      and env.npc.pos[1] < env.w - pad - reach):
                    action = 3 if _DIRS[3][1] * away[1] >= 0 else 4
                else:
                    # boxed into a wall/corner with no room to maneuver: stand
                    # and strike (scrum hits always land) to break the press
                    action = 9
            elif mdir != 0 and float(np.dot(_DIRS[mdir], away)) >= 0.0:
                # too close to advance: stand and strike from blade reach
                action = 9

        _, reward, term, trunc, info = env.step(action)
        self.reward += float(reward)

        # --- detect style from the PLAYER's dynamics ----------------------- #
        combat_info = {
            "player_attack": bool(info.get("player_attacked", False)),
            "player_hit": float(info.get("player_damage_dealt", 0.0)) > 0,
            "damage_dealt": float(info.get("player_damage_dealt", 0.0)),
            "player_hp_frac": info["player_hp"] / env.player_max_hp,
        }
        self.fw.update(FeatureWindow.extract_step(
            env.npc_pos, self.prev_npc, env.player.pos, self.prev_player,
            combat=combat_info))
        feat = self.fw.features()
        if feat is not None:
            self.fh.add(feat)
            if self.fh.ready():
                dlabel, probs = self.det_model.predict(self.fh.as_context())
                if self.declared is not None:
                    # human is deliberately playing a style: the declared lock
                    # is sticky; the detector still runs (boosted) for the
                    # transition counter / HUD, but never overrides the user.
                    probs = self._prior_boost(probs)
                    _, is_trans = self.cusum.update(probs)
                    if is_trans:
                        self.transitions += 1
                        self.fx.flash_with(0.22)
                    self.soft_conf = 0.8
                else:
                    # no assist pressed -> fully inferred detection, with a
                    # soft lock that fires well before CUSUM for responsiveness.
                    lab = int(np.argmax(probs))
                    conf = float(probs[lab])
                    self.soft_streak = self.soft_streak + 1 if conf > 0.60 else 0
                    changed = False
                    prev = self.detected if self.detected is not None else -1
                    if self.soft_streak >= 2 and lab != prev:
                        self.detected = lab
                        self.adapt_events += 1
                        self.soft_conf = max(self.soft_conf, conf)
                        self._record_detection(lab)
                        changed = True
                    dlabel, is_trans = self.cusum.update(probs)
                    if is_trans:
                        self.transitions += 1
                        self.soft_conf = 0.85
                        if self.detected != dlabel:
                            self.detected = dlabel
                            self.adapt_events += 1
                            self._record_detection(dlabel)
                            changed = True
                        self.fx.flash_with(0.22)
                    if changed:
                        self._banner_plan(PLAN_OF_STYLE.get(
                            LABELS[self.detected], "SWAY"))

        self.assist_phase += 1.0

        # --- attack animation triggers (cooldown just reset => a swing) ---- #
        pcd, ncd = env.player.attack_cooldown, env.npc.attack_cooldown
        cd = env.attack_cooldown
        if self.prev_pcd <= 0.0 and pcd >= cd - 1e-9:
            self.player_v.sword.trigger()
            self.player_v.flash(0.25)
            self.sfx.play("whoosh_p")
        if self.prev_ncd <= 0.0 and ncd >= cd - 1e-9:
            self.npc_v.sword.trigger()
            self.sfx.play("whoosh_n")
        self.prev_pcd, self.prev_ncd = pcd, ncd

        # --- hit / kill effects ------------------------------------------- #
        dealt = float(info.get("damage_dealt", 0.0))     # NPC -> player
        taken = float(info.get("damage_taken", 0.0))     # player -> NPC
        if taken > 0:
            self.fx.add_sparks(env.npc.pos, (140, 235, 255), 12)
            self.fx.add_num(env.npc.pos, f"{-int(taken)}", (170, 235, 255))
            self.fx.shake_with(4)
            self.npc_v.flash(0.45)
            self.sfx.play("clink_p")
        if dealt > 0:
            self.fx.add_sparks(env.player.pos, (255, 140, 120), 12)
            self.fx.add_num(env.player.pos, f"{-int(dealt)}", (255, 170, 140))
            self.fx.shake_with(5)
            self.player_v.flash(0.45)
            self.sfx.play("clink_n")

        if info["win"]:
            self.deaths += 1
            self.result = "YOU DIED"
            self.fx.add_burst(env.player.pos, (255, 120, 100), 40)
            self.fx.shake_with(10)
            self.fx.flash_with(0.5)
            self.sfx.play("boom_p")
        elif info["lose"]:
            self.kills += 1
            self.result = "VICTORY" if not self.watch else "NPC DOWN"
            self.fx.add_burst(env.npc.pos, (140, 235, 255), 40)
            self.fx.shake_with(10)
            self.fx.flash_with(0.5)
            self.sfx.play("boom_n")

        if term or trunc:
            if not (info["win"] or info["lose"]) and trunc:
                self.result = "TIME UP"
            self.over = True
            self.over_t = 0.0

        # --- watch-mode personality switch (auto-cycle only) ----------------- #
        if self.watch and self.fixed_style is None and env.step_count == SWITCH_STEP:
            nxt = LABELS[(LABELS.index(env.player.label) + 1) % N]
            env.set_player_strategy(nxt)
            self.fx.flash_with(0.35)
            self.fx.add_num(env.player.pos, f"SWITCH -> {nxt.upper()}", (255, 230, 120))

        self.prev_npc = env.npc_pos.copy()
        self.prev_player = env.player.pos.copy()
        return info

    def _prior_boost(self, probs):
        """Push classifier output toward a hand-held style-assist key."""
        s = self.style_assist
        if s < 1 or s > 4:
            return probs
        p = np.asarray(probs, dtype=np.float64).copy()
        p[s - 1] *= 6.0
        p /= max(p.sum(), 1e-9)
        return p

    def _declare_style(self, idx):
        """Instantly commit the NPC to counter-playing a declared style."""
        self.declared = idx
        if self.detected != idx or self.npc_plan != PLAN_OF_STYLE[LABELS[idx]]:
            self.detected = idx
            self.soft_conf = 0.8
            self.adapt_events += 1
            self._banner_plan(PLAN_OF_STYLE[LABELS[idx]])

    def _record_detection(self, idx):
        """Remember a committed style lock (for the HUD timeline strip)."""
        self.det_history.append(idx)

    def _banner_plan(self, plan):
        """NPC visibly commits to a counter game-plan."""
        self.npc_plan = plan
        self.plan_t = 1.4
        self.plan_flash = 1.0
        style = LABELS[self.detected] if self.detected is not None else None
        col = PLAN_COLOR[plan]
        self.banner_text = f"{plan}  vs  {style.upper()}" if style else plan
        self.banner_sub = PLAN_EXPLAIN.get(plan, "")
        self.banner_color = col
        self.banner_t = 2.0
        self.fx.add_num(self.env.npc.pos, f"{plan}!", col)
        self.fx.flash_with(0.18)
        self.npc_v.flash(0.45)
        self.sfx.play("plan")
        self.sfx.play("lock")

    def apply_menu_style(self, idx):
        """Switch to watch mode and challenge the NPC against a chosen style.
        idx None = auto-cycle showcase; otherwise a LABELS index."""
        self.fixed_style = idx
        self.menu_open = False
        self.reset_episode(watch=True)
        if idx is None:
            self.banner_text = "AUTO-CYCLE"
            self.banner_color = (255, 230, 120)
        else:
            lab = LABELS[idx]
            self.banner_text = f"TESTING: PLAYER = {lab.upper()}"
            self.banner_color = PERSONA_COLOR[lab]
        self.banner_t = 2.0
        self.sfx.play("lock")

    def _draw_menu(self, s):
        """Pause overlay: set the player style the NPC must adapt to."""
        bg = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        bg.fill((6, 8, 12, 205))
        s.blit(bg, (0, 0))
        w, h = 460, 380
        x, y = WIN_W // 2 - w // 2, 60
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((16, 20, 28, 235))
        pygame.draw.rect(panel, (110, 130, 170), panel.get_rect(), 2)
        s.blit(panel, (x, y))

        def center(text, font, color, yy):
            surf = font.render(text, True, color)
            s.blit(surf, (WIN_W // 2 - surf.get_width() // 2, yy))

        title = pygame.font.SysFont("consolas", 22, bold=True)
        sub = pygame.font.SysFont("consolas", 13)
        it = pygame.font.SysFont("consolas", 17, bold=True)
        warn = pygame.font.SysFont("consolas", 13)

        center("PLAYER STYLE  vs  NPC", title, (230, 235, 245), y + 18)
        center("the autopilot player fights in this style;", sub,
               (150, 160, 180), y + 50)
        center("the NPC must DETECT it and ADAPT", sub, (150, 160, 180), y + 68)

        row_y = y + 102
        for i, (name, style, hint) in enumerate(MENU_ITEMS):
            sel = i == self.menu_sel
            col = (255, 235, 130) if sel else (205, 212, 225)
            name_c = PERSONA_COLOR[name.lower()] if style is not None else col
            label = ("->  " if sel else "    ") + name
            surf = it.render(label, True, name_c if style is not None else col)
            s.blit(surf, (x + 34, row_y))
            hs = sub.render(hint, True, (130, 140, 155))
            s.blit(hs, (x + 210, row_y + 4))
            if sel:
                pygame.draw.rect(s, (70, 90, 130), (x + 14, row_y - 3,
                                                    w - 28, 28), 1)
            row_y += 44

        foot = pygame.font.SysFont("consolas", 13)
        center("W/S or arrows select   ENTER/SPACE/1-4 apply   M/ESC close   Q quit",
               foot, (140, 150, 170), row_y + 24)

    # ------------------------------------------------------------------ #
    def draw_hud(self):
        env = self.env
        s = self.screen
        # bottom strip
        pygame.draw.rect(s, (14, 16, 22), pygame.Rect(0, WORLD_PX, WIN_W, HUD_H))
        pygame.draw.line(s, (60, 70, 95), (0, WORLD_PX), (WIN_W, WORLD_PX), 2)

        det = LABELS[self.detected] if self.detected is not None else "waiting..."
        seg = LABELS[self.cusum.segment_label] if self.cusum.segment_label is not None else "waiting..."
        player_lab = env.player.label

        def blit(text, font, color, x, y):
            surf = font.render(text, True, color)
            screen_blit_pos = (int(x), int(y))
            self.screen.blit(surf, screen_blit_pos)

        y1 = WORLD_PX + 8
        blit(f"you:{player_lab:<9s}", self.hud_font, PERSONA_COLOR[player_lab], 12, y1)
        blit(f"npc-sees:{det:<9s} seg:{seg:<9s}", self.hud_font, (200, 210, 225), 150, y1)
        rsurf = self.hud_font.render(f"reward={self.reward:6.0f}", True, (150, 160, 185))
        self.screen.blit(rsurf, (WIN_W - rsurf.get_width() - 12, y1))

        pc = PLAN_COLOR[self.npc_plan]
        y2 = WORLD_PX + 30
        blit(f"npc-plan:{self.npc_plan:<5s}", self.hud_font_s, pc, 12, y2)
        segw = self.hud_font_s.render(f"npc-plan:{self.npc_plan:<5s}", True, pc).get_width()
        fix = " FIXED" if self.fixed_style is not None else ""
        blit(f"mode:{'WATCH' if self.watch else 'VS'}  yours:{player_lab}{fix}  "
             f"adapts={self.adapt_events}  transitions={self.transitions}",
             self.hud_font_s, (255, 230, 120) if self.watch else (185, 195, 210),
             12 + segw + 8, y2)
        assist = ASSIST_NAME.get(self.style_assist, "").lower()
        if self.style_assist:
            blit(f"[{assist} style]",
                 self.hud_font_s, PERSONA_COLOR[player_lab], 12, y2 + 14)

        y3 = WORLD_PX + 52
        ctrl = ("WASD move | SPACE attack | M style-test menu | T watch | R | Q"
                if not self.watch
                else "WATCHING autopilot | 1-4 override | M style menu | T back | Q")
        csurf = self.hud_font_s.render(ctrl, True, (120, 130, 150))
        self.screen.blit(csurf, (WIN_W // 2 - csurf.get_width() // 2, y3))
        if not self.watch:
            kd = self.hud_font_s.render(
                f"K:{self.kills} D:{self.deaths}", True,
                (150, 235, 160) if self.kills >= self.deaths else (235, 120, 110))
            self.screen.blit(kd, (WIN_W - kd.get_width() - 12, y3))

        # in-arena top bars: YOU (left) / NPC (right) with pips
        self._hp_panel(12, 12, "YOU", env.player.hp, env.player_max_hp,
                       env.player.attack_cooldown, env.attack_cooldown,
                       (235, 90, 80))
        self._hp_panel(WIN_W - 190, 12, "NPC", env.npc.hp, env.npc_max_hp,
                       env.npc.attack_cooldown, env.attack_cooldown,
                       (80, 150, 230))

        # episode result banner
        if self.over and self.result:
            bg = pygame.Surface((WIN_W, WORLD_PX), pygame.SRCALPHA)
            bg.fill((8, 10, 14, 120))
            s.blit(bg, (0, 0))
            label = self.big_font.render(self.result, True, (255, 255, 255))
            sub = self.hud_font_s.render(
                "R = restart", True, (200, 205, 215))
            s.blit(label, (WIN_W // 2 - label.get_width() // 2, WORLD_PX // 2 - 40))
            s.blit(sub, (WIN_W // 2 - sub.get_width() // 2, WORLD_PX // 2))

    def _hp_panel(self, x, y, name, hp, hp_max, cd, cd_max, color):
        s = self.screen
        cd_max = int(cd_max)
        w = 178
        bar_h = 14
        pygame.draw.rect(s, (8, 10, 14), (x - 4, y - 4, w + 8, bar_h + 26))
        nsurf = self.hud_font_s.render(name, True, (220, 225, 235))
        s.blit(nsurf, (x, y))
        frac = max(0.0, min(1.0, hp / max(hp_max, 1)))
        col = (120, 220, 130)
        if frac < 0.5:
            col = (245, 205, 90)
        if frac < 0.25:
            col = (235, 80, 70)
        pygame.draw.rect(s, (45, 50, 62), (x + 34, y, w - 34, bar_h))
        pygame.draw.rect(s, col, (x + 34, y, int((w - 34) * frac), bar_h))
        pygame.draw.rect(s, (200, 205, 220), (x + 34, y, w - 34, bar_h), 1)
        # cooldown pips
        for i in range(cd_max):
            px = x + 34 + i * ((w - 34) // cd_max)
            active = (cd <= 0) or (i < cd_max - int(cd))
            c = (255, 235, 150) if active else (60, 65, 80)
            pygame.draw.rect(s, c, (px, y + bar_h + 4, (w - 34) // cd_max - 2, 5))
        hpnum = self.hud_font_s.render(f"{int(hp)}/{hp_max}", True, (255, 255, 255))
        s.blit(hpnum, (x + w - 66, y + bar_h + 10))

    # ------------------------------------------------------------------ #
    def tick(self):
        """Advance sim time as real time demands, then render."""
        now = time.perf_counter()
        rdt = min(0.08, max(0.0, now - self.last))
        self.last = now
        self.sim_acc += rdt
        steps = 0
        while (self.sim_acc >= SIM_DT and steps < 4 and (not self.over)
               and not self.menu_open):
            self.run_step()
            self.sim_acc -= SIM_DT
            steps += 1
        # refresh visuals even while over (fx fade)
        self.player_v.update(rdt, self.env.player.pos, self.env.player.facing,
                             speed=float(np.linalg.norm(self.env.player.vel)))
        self.npc_v.update(rdt, self.env.npc.pos, self.env.npc.facing,
                          speed=float(np.linalg.norm(self.env.npc.vel)))
        self.fx.update(rdt)
        self.plan_t = max(0.0, self.plan_t - rdt)
        self.plan_flash = max(0.0, self.plan_flash - rdt * 2.5)
        self.banner_t = max(0.0, self.banner_t - rdt)
        if self.over:
            self.over_t += rdt
            if self.watch and self.over_t > 1.4:
                self.reset_episode(watch=True)

    # ------------------------------------------------------------------ #
    def render(self):
        s = self.screen
        sx = sy = 0
        if self.fx.shake > 0:
            rng = np.random.default_rng()
            sx = int(rng.normal() * self.fx.shake)
            sy = int(rng.normal() * self.fx.shake)
        s.fill((10, 12, 18))
        s.blit(self.arena, (sx, sy))

        # low-hp danger vignette
        hp_frac = self.env.player.hp / max(self.env.player_max_hp, 1)
        if hp_frac < 0.35 and not self.over:
            p = 0.5 * (1 - hp_frac / 0.35) * (0.6 + 0.4 * np.sin(self.frame_n * 0.18))
            vg = pygame.Surface((WORLD_PX, WORLD_PX), pygame.SRCALPHA)
            wth = int(30 + 26 * p)
            pygame.draw.rect(vg, (200, 28, 24, int(150 * p)),
                             (0, 0, WORLD_PX, WORLD_PX), wth)
            s.blit(vg, (0, 0))

        # flash overlay
        if self.fx.flash > 0:
            fl = pygame.Surface((WORLD_PX, WORLD_PX), pygame.SRCALPHA)
            fl.fill((255, 255, 255, int(120 * self.fx.flash)))
            s.blit(fl, (0, 0))

        self.player_v.draw(s, self.scale, self.env.player.pos,
                           self.env.player.facing,
                           persona_c=PERSONA_COLOR[self.env.player.label])
        npc_style = LABELS[self.detected] if self.detected is not None else None
        self.npc_v.draw(s, self.scale, self.env.npc.pos,
                        self.env.npc.facing,
                        persona_c=PERSONA_COLOR[npc_style] if npc_style else None)

        # plan pulse ring around the NPC while it commits to a game-plan
        if self.plan_t > 0 and npc_style:
            npcx, npcy = self.to_px(self.env.npc.pos)
            pr = int(12 + (1.4 - self.plan_t) * 26)
            a = int(120 * min(1.0, self.plan_t * 2.5))
            ring = pygame.Surface((pr * 2, pr * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*PLAN_COLOR[self.npc_plan], a),
                               (pr, pr), pr, 2)
            s.blit(ring, (npcx - pr, npcy - pr))

        self.fx.draw(s)
        self._draw_aim(s)
        self._draw_plan_badge(s)
        self._draw_plan_chevrons(s)
        self._draw_detect_strip(s)
        self._draw_banner(s)
        self.draw_hud()
        if self.menu_open:
            self._draw_menu(s)
        pygame.display.flip()

    def _draw_banner(self, s):
        """Big center-screen announcement when the NPC commits to a plan."""
        if not self.banner_text or self.banner_t <= 0:
            return
        k = min(1.0, self.banner_t / 1.5)
        a = int(235 * k)
        font = pygame.font.SysFont("consolas", 30, bold=True)
        sub_font = pygame.font.SysFont("consolas", 15)
        main = font.render(self.banner_text, True, self.banner_color)
        main.set_alpha(a)
        w = max(main.get_width() + 44, 320)
        h = 78 if self.banner_sub else 58
        bg = pygame.Surface((w, h), pygame.SRCALPHA)
        bg.fill((10, 12, 18, int(180 * k)))
        pygame.draw.rect(bg, (*self.banner_color, a), (0, 0, w, h), 2)
        bx = WIN_W // 2 - w // 2
        by = WIN_H // 2 - 120 + (1.0 - k) * 24
        s.blit(bg, (bx, by))
        s.blit(main, (WIN_W // 2 - main.get_width() // 2,
                      by + 10))
        if self.banner_sub:
            sb = sub_font.render(self.banner_sub, True, (210, 220, 235))
            sb.set_alpha(a)
            s.blit(sb, (WIN_W // 2 - sb.get_width() // 2, by + 48))

    def _draw_aim(self, s):
        """Auto-target indicator: pulsing ring on the NPC once your sword is
        ready, green when in range. A faint line shows the auto-aimed swing."""
        if self.watch:
            return
        keys = pygame.key.get_pressed()
        attacking = bool(keys[pygame.K_SPACE]) or bool(pygame.mouse.get_pressed()[0])
        env = self.env
        player_reach = float(env.combat.get("player_reach", env.attack_range))
        d = float(np.linalg.norm(env.npc.pos - env.player.pos))
        in_range = d < player_reach
        npcx, npcy = self.to_px(env.npc.pos)
        cd = float(env.player.attack_cooldown)
        cd_max = float(env.attack_cooldown)
        if cd <= 0.0:
            t = self.frame_n * 0.09
            r = 12 + int(3 * np.sin(t))
            c = (140, 255, 150) if in_range else (225, 230, 240)
            ring = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*c, 200), (r, r), r, 2)
            pygame.draw.circle(ring, (*c, 255), (r, r), 2)
            s.blit(ring, (npcx - r, npcy - r))
        elif attacking:
            # press registered while the sword recharges: show a shrinking
            # arc so the 0.6s reload is visible instead of a silent miss
            fill = 1.0 - max(0.0, min(1.0, cd / cd_max))
            r = 14
            arc = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            start = math.pi * 0.5
            sweep = 2.0 * math.pi * fill
            pygame.draw.arc(arc, (235, 190, 90, 235),
                            (0, 0, r * 2, r * 2), start, start + sweep, 3)
            pygame.draw.circle(arc, (235, 190, 90, 120), (r, r), r, 1)
            s.blit(arc, (npcx - r, npcy - r))
        if attacking:
            line = pygame.Surface((WIN_W, WORLD_PX), pygame.SRCALPHA)
            pygame.draw.line(line, (255, 255, 255, 46),
                             self.to_px(env.player.pos),
                             self.to_px(env.npc.pos), 2)
            s.blit(line, (0, 0))

    def to_px(self, p):
        return int(p[0] * self.scale), int(p[1] * self.scale)

    # ------------------------------------------------------------------ #
    def _draw_plan_badge(self, s):
        """Persistent floating tag over the NPC: current counter-plan + the
        style it is adapting to. Pops on a fresh commit (`plan_flash`)."""
        plan = self.npc_plan
        if plan is None:
            return
        npcx, npcy = self.to_px(self.env.npc.pos)
        style = LABELS[self.detected] if self.detected is not None else "?"
        pc = PLAN_COLOR[plan]
        sc = PERSONA_COLOR[style] if style != "?" else (150, 160, 180)
        pop = 1.0 + 0.30 * self.plan_flash
        font = pygame.font.SysFont("consolas", 15, bold=True)
        sub_font = pygame.font.SysFont("consolas", 12)
        main = font.render(plan, True, pc)
        sub = sub_font.render(f"ADAPTS TO {style.upper()}", True, sc)
        w = max(main.get_width() + 40, sub.get_width() + 32)
        h = int(44 * pop)
        bx = int(npcx - w // 2)
        by = int(npcy - self.npc_v.r * 3.2 - 18 * pop)
        if by < 4:
            by = 4
        bh = pygame.Surface((w, h), pygame.SRCALPHA)
        bh.fill((10, 14, 22, 205))
        pygame.draw.rect(bh, (*pc, 240), (0, 0, w, h), 2)
        glow = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
        pygame.draw.rect(glow, (*pc, int(60 * self.plan_flash)),
                         (0, 0, w + 8, h + 8), 3)
        s.blit(glow, (bx - 4, int(by - 4)))
        s.blit(bh, (bx, int(by)))
        s.blit(main, (bx + w // 2 - main.get_width() // 2,
                      by + 4))
        s.blit(sub, (bx + w // 2 - sub.get_width() // 2,
                     by + h - 17))

    def _draw_plan_chevrons(self, s):
        """Direction arrows beside the NPC making the plan intent legible:
        KITE retreats (arrow away), HUNT/PRESS chase, SWAY shows a pulsing dot."""
        plan = self.npc_plan
        if plan is None:
            return
        npcx, npcy = self.to_px(self.env.npc.pos)
        to_p = self.env.player.pos - self.env.npc.pos
        dn = float(np.linalg.norm(to_p))
        if dn < 1e-6:
            return
        to_pn = to_p / dn
        pc = PLAN_COLOR[plan]
        r = self.npc_v.r
        t = self.frame_n * 0.15
        if plan == "KITE":
            d = -to_pn
        elif plan in ("HUNT", "PRESS"):
            d = to_pn
        else:
            # SWAY: small pulsing ring, no arrow
            ph = 0.5 + 0.5 * np.sin(t)
            qr = int(10 + 3 * ph)
            ring = pygame.Surface((qr * 2, qr * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring, (*pc, int(160 * ph)),
                               (qr, qr), qr, 2)
            s.blit(ring, (npcx - qr, npcy - r - 8 - qr))
            return
        pd = np.array([-d[1], d[0]])
        for i in range(2):
            k = 1.0 + 0.45 * i
            base = np.array([npcx, npcy]) + d * (r + 10) * k
            a = 150 + int(105 * (0.5 + 0.5 * np.sin(t + i)))
            ar = pygame.Surface((26, 26), pygame.SRCALPHA)
            cx0, cy0 = 13, 13
            tip = (cx0 + d[0] * 8, cy0 + d[1] * 8)
            back0 = (cx0 - d[0] * 8 - pd[0] * 4, cy0 - d[1] * 8 - pd[1] * 4)
            back1 = (cx0 - d[0] * 8 + pd[0] * 4, cy0 - d[1] * 8 + pd[1] * 4)
            pygame.draw.lines(ar, (*pc, a), False, [back0, tip, back1], 3)
            sx = int(base[0]) - 13
            sy = int(base[1]) - 13
            if 0 <= sx <= WIN_W and 0 <= sy <= WORLD_PX:
                s.blit(ar, (sx, sy))

    def _draw_detect_strip(self, s):
        """Timeline of recent committed detections at the arena's top edge:
        one colored cell per style lock, newest on the right; the live
        confidence bar sits underneath. Makes adaptation visible over time."""
        n = len(self.det_history)
        if n == 0:
            return
        env = self.env
        cell = 14
        gap = 4
        total = n * (cell + gap) - gap
        x0 = (WIN_W - total) // 2
        y0 = 6
        pygame.draw.rect(s, (14, 16, 24), (x0 - 9, y0 - 3, total + 18, 30))
        for i, lab in enumerate(self.det_history):
            c = PERSONA_COLOR[LABELS[lab]]
            x = x0 + i * (cell + gap)
            cur = (i == n - 1 and (self.detected == lab or True))
            box = pygame.Surface((cell, cell), pygame.SRCALPHA)
            box.fill((*c, 210))
            pygame.draw.rect(box, (255, 255, 255), (0, 0, cell, cell), 1 if cur else 0)
            s.blit(box, (x, y0))
        bar_w = min(220, max(60, total))
        bx = (WIN_W - bar_w) // 2
        by = y0 + cell + 7
        bg = pygame.Surface((bar_w, 6), pygame.SRCALPHA)
        bg.fill((40, 46, 60, 220))
        s.blit(bg, (bx, by))
        conf = max(0.0, min(1.0, self.soft_conf))
        fw = max(2, int(bar_w * conf))
        fg = pygame.Surface((fw, 6), pygame.SRCALPHA)
        col = (150, 235, 170) if conf >= 0.7 else (235, 200, 90)
        fg.fill((*col, 240))
        s.blit(fg, (bx, by))
        lbl = pygame.font.SysFont("consolas", 10).render(
            f"detect lock {int(conf * 100)}%", True, (200, 210, 225))
        s.blit(lbl, (bx + bar_w + 6, by - 1))

    def run(self):
        while self.running:
            keys = pygame.key.get_pressed()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.running = False
                elif e.type == pygame.KEYDOWN:
                    k = e.key
                    if k == pygame.K_q:
                        self.running = False
                    elif k == pygame.K_ESCAPE:
                        if self.menu_open:
                            self.menu_open = False
                        else:
                            self.running = False
                    elif k == pygame.K_m:
                        self.menu_open = not self.menu_open
                        if self.menu_open:
                            self.menu_sel = (0 if self.fixed_style is None
                                             else self.fixed_style + 1)
                    elif self.menu_open:
                        if k in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_KP_ENTER):
                            self.apply_menu_style(MENU_ITEMS[self.menu_sel][1])
                        elif k in (pygame.K_UP, pygame.K_w):
                            self.menu_sel = (self.menu_sel - 1) % len(MENU_ITEMS)
                        elif k in (pygame.K_DOWN, pygame.K_s):
                            self.menu_sel = (self.menu_sel + 1) % len(MENU_ITEMS)
                        elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                            self.apply_menu_style(k - pygame.K_1)
                    elif k == pygame.K_r:
                        self.reset_episode()
                    elif k == pygame.K_t:
                        self.reset_episode(watch=not self.watch)
                    elif k in ASSIST_KEYS:
                        s = ASSIST_KEYS[k]
                        if self.watch:
                            self.env.set_player_strategy(LABELS[s - 1])
                            self.fx.add_num(self.env.player.pos,
                                            f"YOU: {LABELS[s - 1].upper()}",
                                            PERSONA_COLOR[LABELS[s - 1]])
                            self.fx.flash_with(0.2)
                        else:
                            self.style_assist = 0 if self.style_assist == s else s
                            if self.style_assist:
                                self._declare_style(s - 1)
                            else:
                                self.declared = None
                                self.soft_streak = 0
                                if self.detected is not None:
                                    self.soft_conf = 0.5
            lmb = bool(pygame.mouse.get_pressed()[0])
            if not self.menu_open:
                self.handle_input(keys, keys[pygame.K_SPACE] or lmb)
            self.tick()
            self.render()
            self.frame_n += 1
            if self.shot and self.frame_n == self.shot_at:
                pygame.image.save(self.screen, self.shot)
                self.running = False
            if self.frames and self.frame_n >= self.frames:
                self.running = False
            self.clock.tick(FPS)
        pygame.quit()


def main(argv):
    frames = None
    shot = None
    shot_at = None
    model_path = DEFAULT_MODEL if os.path.exists(DEFAULT_MODEL) else FALLBACK_MODEL
    rest = list(argv)
    if rest and "--frames" in rest:
        i = rest.index("--frames")
        frames = int(rest[i + 1])
        del rest[i:i + 2]
    if rest and "--shot" in rest:
        i = rest.index("--shot")
        shot = rest[i + 1]
        del rest[i:i + 2]
    if rest and "--at" in rest:
        i = rest.index("--at")
        shot_at = int(rest[i + 1])
        del rest[i:i + 2]
    if rest:
        model_path = rest[0]
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}")
        return 1
    cfg = load_config()
    print(f"model: {model_path} -- loading combat config (fun-tuned) ...")
    g = Game(cfg, model_path, frames=frames, shot_at=shot_at)
    g.shot = shot
    g.run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))