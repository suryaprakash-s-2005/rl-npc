"""Interactive human-vs-NPC combat demo.

You control the player (red) with WASD / arrow keys; hold SPACE to swing your
sword (facing auto-aims at the NPC when you attack in place). Release all keys
to stop. You and the NPC both have HP (HP-20); land hits to drain the NPC's HP
and kill it before it kills you.

The NPC (blue) fights with a policy trained on strategy-transition-aware
experience sharing, conditioned on the strategy the online detector infers from
YOUR movement/attack dynamics. Mid-episode the player personality switches to a
paired partner personality -- watch the NPC adapt.

HUD shows: detected player strategy + segment, HP bars, damage dealt/taken,
kill/death count, transitions, reward.

Controls:
  WASD / arrows : steer the player
  SPACE         : attack (swing toward the NPC)
  R             : reset the episode (new random personality)
  ESC / Q       : quit

Usage:
    python -m demo.play [model_path]
        (model_path defaults to data/npc_partitioned-gt-seed0.pt)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from src.config import load_config
from src.env.chase_env import ChaseEnv, N_ACTIONS
from src.env.features import FeatureWindow
from src.rl.dqn_agent import DQNAgent
from src.strategy.detector import FeatureHistory, StrategyClassifier
from src.strategy.transitions import CusumTransitionDetector

LABELS = ["aggressive", "defensive", "evasive", "balanced"]
N = len(LABELS)

FPS = 60
WORLD_PX = 600
MARGIN = 12
FOOTER = 92


def draw_world(screen, scale, width, height):
    pygame.draw.rect(screen, (20, 24, 32), pygame.Rect(0, 0, width, height))
    for i in range(1, int(width / scale)):
        x = int(i * scale)
        pygame.draw.line(screen, (36, 42, 54), (x, 0), (x, height), 1)
    for j in range(1, int(height / scale)):
        y = int(j * scale)
        pygame.draw.line(screen, (36, 42, 54), (0, y), (width, y), 1)


def main(argv):
    cfg = load_config()
    model_path = argv[0] if argv else os.path.join("data", "npc_partitioned-gt-seed0.pt")
    if not os.path.exists(model_path):
        print(f"model not found: {model_path}")
        print("train one first, e.g.: python -m src.train --rl.replay partitioned")
        return 1

    import torch
    # load detector
    det_model = StrategyClassifier(cfg)
    ck_det = torch.load(cfg["detector"]["model_path"], map_location="cpu")
    det_model.load_state_dict(ck_det["state_dict"])
    det_model.eval()

    # auto-detect whether the NPC model conditions on the strategy label by
    # reading its input dimension (base combat obs = 14; +4 one-hot when cond-on)
    ck = torch.load(model_path, map_location="cpu")
    in_dim = int(ck["online"]["net.0.weight"].shape[1])
    cfg["rl"]["cond_on_strategy"] = in_dim > 14
    print(f"model: {model_path}  cond_on_strategy={cfg['rl']['cond_on_strategy']} "
          f"(obs dim {in_dim})")

    env = ChaseEnv(cfg, seed=int(cfg["seed"]))
    agent = DQNAgent(cfg, env.obs_dim(), N_ACTIONS)
    agent.load(model_path)

    pygame.init()
    screen = pygame.display.set_mode((WORLD_PX, WORLD_PX + FOOTER))
    pygame.display.set_caption("Human vs NPC -- strategy-aware combat")
    font = pygame.font.SysFont("consolas", 15)
    clock = pygame.time.Clock()

    scale = WORLD_PX / env.w

    def world_to_px(pos):
        return int(pos[0] * scale), int(pos[1] * scale)

    # ---- per-episode state ------------------------------------------- #
    fw = FeatureWindow(cfg["strategy"]["feature_window"])
    fh = FeatureHistory(cfg["detector"]["context_len"])
    cusum = CusumTransitionDetector(cfg, N)
    prev_npc = None
    prev_player = None
    runs = 0
    kills = 0
    deaths = 0
    episode_reward = 0.0
    detected_label = None
    transitions = 0
    over = False
    info = {"dist": 0.0, "win": False, "lose": False, "player_hp": 0.0, "npc_hp": 0.0}
    episode_idx = 0

    def reset_episode():
        nonlocal fw, fh, cusum, prev_npc, prev_player, episode_reward, over, transitions, runs, info, detected_label, episode_idx
        label = LABELS[episode_idx % N]
        episode_idx += 1
        env.reset(player_strategy=label)
        fw = FeatureWindow(cfg["strategy"]["feature_window"])
        fh = FeatureHistory(cfg["detector"]["context_len"])
        cusum = CusumTransitionDetector(cfg, N)
        prev_npc = env.npc_pos.copy()
        prev_player = env.player.pos.copy()
        episode_reward = 0.0
        over = False
        transitions = 0
        detected_label = None
        runs += 1
        info = {"dist": 0.0, "win": False, "lose": False, "player_hp": env.player.hp, "npc_hp": env.npc.hp}
        # force player to manual with zero velocity (no movement until keys held)
        env.player.set_manual_velocity([0.0, 0.0])

    reset_episode()

    running = True
    while running:
        # ---- events --------------------------------------------------- #
        keys = pygame.key.get_pressed()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_r:
                    reset_episode()

        # manual velocity from held keys (WASD + arrows); SPACE swings the sword
        ux = (int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) -
              int(keys[pygame.K_LEFT] or keys[pygame.K_a]))
        uy = (int(keys[pygame.K_DOWN] or keys[pygame.K_s]) -
              int(keys[pygame.K_UP] or keys[pygame.K_w]))
        attack = bool(keys[pygame.K_SPACE])
        if ux or uy:
            speed = env.env_cfg["npc_max_speed"] * 0.9
            mv = np.array([ux, uy], dtype=np.float64)
            env.player.set_manual_velocity(mv / max(np.linalg.norm(mv), 1e-6) * speed, attack=attack)
        elif attack:
            # attack in place: zero velocity keeps facing <- step() auto-aims at NPC
            env.player.set_manual_velocity([0.0, 0.0], attack=attack)
        else:
            # no keys held: player stays still
            env.player.set_manual_velocity([0.0, 0.0])

        if not over:
            # strategy detection from YOUR movement dynamics
            if prev_npc is not None:
                fw.update(FeatureWindow.extract_step(
                    env.npc_pos, prev_npc, env.player.pos, prev_player))
                feat = fw.features()
                if feat is not None:
                    fh.add(feat)
                    if fh.ready():
                        dlabel, probs = det_model.predict(fh.as_context())
                        dlabel, is_trans = cusum.update(probs)
                        detected_label = dlabel
                        if is_trans:
                            transitions += 1

            # NPC action conditioned on detected strategy
            cond = bool(cfg["rl"].get("cond_on_strategy", False))
            label_id = detected_label if detected_label is not None else LABELS.index("balanced")
            base = env._obs()
            obs = env.observed_signal(base, label_id, cond)
            action = agent.greedy(obs)
            next_obs, reward, term, trunc, info = env.step(action)
            episode_reward += float(reward)
            if info["win"]:
                deaths += 1
                over = True
            elif info["lose"]:
                kills += 1
                over = True

        prev_npc = env.npc_pos.copy()
        prev_player = env.player.pos.copy()

        # ---- render --------------------------------------------------- #
        draw_world(screen, scale, WORLD_PX, WORLD_PX)

        a = world_to_px(env.player.pos)
        b = world_to_px(env.npc_pos)

        # distance line
        pygame.draw.line(screen, (90, 100, 120), a, b, 1)

        # player (red) + heading
        pygame.draw.circle(screen, (230, 80, 80), a, 10)
        pv = env.player.vel
        if np.linalg.norm(pv) > 1e-3:
            h = pv / np.linalg.norm(pv)
            pygame.draw.line(screen, (255, 200, 200), a,
                             (int(a[0] + h[0] * 18), int(a[1] + h[1] * 18)), 2)

        # npc (blue) + heading
        pygame.draw.circle(screen, (80, 160, 230), b, 9)
        nv = env.npc_vel
        if np.linalg.norm(nv) > 1e-3:
            hn = nv / np.linalg.norm(nv)
            pygame.draw.line(screen, (180, 220, 255), b,
                             (int(b[0] + hn[0] * 18), int(b[1] + hn[1] * 18)), 2)

        # HP bars above actors
        def hp_bar(center, frac, color):
            bw, bh = 44, 5
            bx = int(center[0] - bw / 2)
            by = max(2, int(center[1] - 22))
            pygame.draw.rect(screen, (40, 44, 54), (bx, by, bw, bh))
            fw_ = max(0, int(bw * min(1.0, max(0.0, frac))))
            if fw_:
                pygame.draw.rect(screen, color, (bx, by, fw_, bh))

        hp_bar(a, env.player.hp / max(env.player_max_hp, 1e-6), (240, 90, 90))
        hp_bar(b, env.npc.hp / max(env.npc_max_hp, 1e-6), (90, 180, 245))

        # border
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, WORLD_PX, WORLD_PX), 2)

        # kill / death flash
        if info["lose"] or info["win"]:
            s = pygame.Surface((WORLD_PX, WORLD_PX), pygame.SRCALPHA)
            s.fill((120, 255, 120, 60) if info["lose"] else (255, 80, 80, 60))
            screen.blit(s, (0, 0))

        # ---- HUD ------------------------------------------------------ #
        # dark backdrop strip + divider under the arena
        pygame.draw.rect(screen, (16, 19, 26), pygame.Rect(0, WORLD_PX, WORLD_PX, FOOTER))
        pygame.draw.line(screen, (70, 80, 100), (0, WORLD_PX), (WORLD_PX, WORLD_PX), 1)

        hy = WORLD_PX + 8
        det = LABELS[detected_label] if detected_label is not None else "waiting..."
        seg = LABELS[cusum.segment_label] if cusum.segment_label is not None else "waiting..."
        praw = env.player.label
        ph = env.player.hp
        nh = env.npc.hp
        pdmg = info.get("damage_dealt", 0.0)
        ptake = info.get("damage_taken", 0.0)
        lines = [
            f"you:{praw:<10s} npc-sees:{det:<10s} segment:{seg:<10s} reward={episode_reward:6.1f}",
            f"hp  you={ph:3.0f}/{env.player_max_hp:.0f}  npc={nh:3.0f}/{env.npc_max_hp:.0f}  "
            f"dmg deal={pdmg:.0f} take={ptake:.0f}",
            f"kills={kills}  deaths={deaths}  transitions={transitions}  dist={info.get('dist',0.0):5.1f}",
            "WASD/arrows=move  SPACE=attack  R=reset  Q=quit",
        ]
        for i, ln in enumerate(lines):
            surf = font.render(ln, True, (220, 225, 235))
            screen.blit(surf, (MARGIN, hy + i * 20))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print(f"demo closed -- you killed {kills} NPC(s), the NPC killed you {deaths} "
          f"time(s) across {runs} reset(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))