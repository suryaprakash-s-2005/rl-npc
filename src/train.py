"""Main RL training loop for the NPC.

Usage:
    python -m src.train [--rl.replay partitioned] [--rl.episodes 150] [--seed 0]

Tracks strategy labels in two modes:
  - groundtruth : uses the player's actual strategy (isolates RL effects)
  - detector    : uses the online classifier + CUSUM transition detector

Transitions are tagged per step; the StrategyPartitionedReplayMemory uses the
current detected strategy to prioritize sampling.
"""

import json
import os
import time

import numpy as np

from .config import load_config
from .env.chase_env import ChaseEnv, N_ACTIONS
from .env.features import FeatureWindow
from .env.strategy import LABEL_TO_ID
from .rl.dqn_agent import DQNAgent
from .rl.replay import ReplayMemory, StrategyPartitionedReplayMemory
from .strategy.detector import FeatureHistory, StrategyClassifier
from .strategy.transitions import CusumTransitionDetector

LABELS = ["aggressive", "defensive", "evasive", "balanced"]


def make_replay(cfg):
    rl = cfg["rl"]
    kind = rl["replay"]
    if kind == "single":
        replay = ReplayMemory(rl["buffer_capacity"])
        labels = None
    else:
        replay = StrategyPartitionedReplayMemory(cfg, rl["buffer_capacity"], len(LABELS))
        labels = LABELS
    return replay, labels


def load_strategy_source(cfg, device="cpu"):
    """Return a callable that yields (label_id, probs) per step given a
    FeatureHistory & CUSUM detector, or None (groundtruth mode)."""
    source = cfg["rl"].get("strategy_source", "groundtruth")
    if source == "groundtruth":
        return None
    path = cfg["detector"]["model_path"]
    if not os.path.exists(path):
        print(f"WARNING detector model not found at {path}; using groundtruth")
        return None
    model = StrategyClassifier(cfg).to(device)
    ck = np.load  # placeholder to keep import simple
    import torch

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def run_episode(cfg, env, agent, replay, current_label, step_counter,
                strategy_model=None, groundtruth=True, epsilon_mode=None,
                switch_label=None, switch_step=None):
    """Run one episode; returns (total_reward, caught, len, transitions,
    detected_labels_hist).

    If switch_label/switch_step are set, the player's personality changes once
    mid-episode (the strategy-transition scenario), so the policy must adapt.
    """
    env.reset(player_strategy=current_label, params_jitter=float(cfg["rl"].get("params_jitter", 0.0)))
    fw = FeatureWindow(cfg["strategy"]["feature_window"])
    fh = FeatureHistory(cfg["detector"]["context_len"]) if not groundtruth else None
    cusum = CusumTransitionDetector(cfg, len(LABELS)) if not groundtruth else None

    prev_npc = env.npc_pos.copy()
    prev_player = env.player.pos.copy()
    prev_info = {}
    total_reward = 0.0
    caught = False
    transitions = 0
    det_labels = []
    ep_steps = 0

    for _step in range(cfg["env"]["episode_len"]):
        cond = bool(cfg["rl"].get("cond_on_strategy", False))

        # within-episode strategy transition (adaptation scenario)
        if switch_label is not None and switch_step is not None and _step == switch_step:
            env.set_player_strategy(switch_label)
            current_label = switch_label

        # resolve the strategy label for this step (detected or ground truth)
        label_id = LABEL_TO_ID[current_label]
        if not groundtruth:
            fw.update(FeatureWindow.extract_step(
                env.npc_pos, prev_npc, env.player.pos, prev_player,
                combat={
                    "player_attack": prev_info.get("player_attacked", False),
                    "player_hit": prev_info.get("player_damage_dealt", 0.0) > 0,
                    "damage_dealt": prev_info.get("player_damage_dealt", 0.0),
                    "player_hp_frac": prev_info.get("player_hp", 1.0) / (env.player.max_hp or 1.0),
                }))
            feat = fw.features()
            det_label = None
            if feat is not None and fh is not None:
                fh.add(feat)
                if fh.ready():
                    _, probs = strategy_model.predict(fh.as_context())
                    det_label, is_trans = cusum.update(probs)
                    if is_trans:
                        transitions += 1
                    label_id = det_label if det_label is not None else label_id
            det_labels.append(label_id)

        base_obs = env._obs()
        obs = env.observed_signal(base_obs, label_id, cond)
        epsilon = agent.epsilon(step_counter[0]) if epsilon_mode is None else epsilon_mode
        action = agent.act(obs, step_counter[0], epsilon=epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = env.observed_signal(next_obs, label_id, cond)
        total_reward += reward
        caught = caught or bool(info["caught"])
        done = terminated or truncated
        ep_steps += 1

        replay.add((obs, action, float(reward), next_obs, 1.0 if done else 0.0),
                   label_id)

        # learn every `learn_every` steps once the buffer holds enough
        learn_every = max(1, int(cfg["rl"].get("learn_every", 2)))
        step_counter[0] += 1
        if len(replay) >= int(cfg["rl"]["batch_size"]) and (step_counter[0] % learn_every == 0):
            batch = replay.sample(cfg["rl"]["batch_size"], current_strategy_id=label_id)
            if batch:
                agent.update(batch)

        prev_npc = env.npc_pos.copy()
        prev_player = env.player.pos.copy()
        prev_info = info

        if done:
            break

    return total_reward, caught, ep_steps, transitions, det_labels


def seed_expert_demos(cfg, env, replay, labels, per_strategy=8):
    """Pre-fill the replay with expert demonstrations.

    An expert "chaser+fighter" moves greedily toward the player and attacks
    with its sword whenever the player is within range. Every seeding episode
    ends in a kill, so the buffer always contains positive terminal rewards for
    all four strategies. The same demonstrations are added to single and
    partitioned memories alike so the RL comparison stays fair.
    """
    from src.env.chase_env import _DIRS

    cond = bool(cfg["rl"].get("cond_on_strategy", False))
    attack_range = float(cfg["combat"]["attack_range"])
    jitter = float(cfg["rl"].get("params_jitter", 0.0))
    added = 0
    for label in labels:
        for _ in range(per_strategy):
            env.reset(player_strategy=label, params_jitter=jitter)
            label_id = LABEL_TO_ID[label]
            for _ in range(int(cfg["env"]["episode_len"])):
                base = env._obs()
                obs = env.observed_signal(base, label_id, cond)
                dvec = env.player.pos - env.npc_pos
                dn = float(np.linalg.norm(dvec))
                dirn = dvec / dn if dn > 1e-6 else np.zeros(2)
                best, best_dot = 0, float("-inf")
                for a in range(len(_DIRS)):
                    dot = float(np.dot(dirn, _DIRS[a]))
                    if a != 0 and dot > best_dot:
                        best_dot, best = dot, a
                action = best if dn > attack_range else best + len(_DIRS)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = env.observed_signal(next_obs, label_id, cond)
                done = terminated or truncated
                replay.add((obs, action, float(reward), next_obs, 1.0 if done else 0.0), label_id)
                added += 1
                if done:
                    break
    return added


def episode_strategy(cfg, ep_idx):
    """Player strategy for a given episode index (cyclic, forces switches)."""
    switch_every = int(cfg["rl"]["switch_every"])
    if switch_every > 0:
        group = ep_idx // switch_every
        return LABELS[group % len(LABELS)]
    return "balanced"


def train_with_config(cfg, silent=False):
    t0 = time.time()
    rl = cfg["rl"]
    env = ChaseEnv(cfg, seed=cfg["seed"])
    agent = DQNAgent(cfg, env.obs_dim(), N_ACTIONS)
    replay, _ = make_replay(cfg)
    strategy_model = load_strategy_source(cfg)
    groundtruth = strategy_model is None

    step_counter = [0]
    history = []
    episode_flags = []

    seed_k = int(rl.get("seed_demos", 8))
    if seed_k > 0 and replay is not None:
        n_demos = seed_expert_demos(cfg, env, replay, LABELS, per_strategy=seed_k)
        if not silent:
            print(f"seeded {n_demos} expert transitions")

    in_switch = bool(rl.get("in_episode_switch", False))
    switch_step = int(rl.get("switch_step", 0))
    switch_pairs = [(LABELS[i], LABELS[(i + 1) % len(LABELS)]) for i in range(len(LABELS))]

    for ep in range(int(rl["episodes"])):
        label = episode_strategy(cfg, ep)
        env.set_player_strategy(label)
        sw_label = sw_step = None
        if in_switch and switch_step > 0:
            # each episode contains one mid-episode personality transition;
            # cycle through the four switch pairs so the policy must adapt
            start, end = switch_pairs[ep % len(switch_pairs)]
            label = start
            sw_label, sw_step = end, switch_step
        total_reward, caught, steps, transitions, det_labels = run_episode(
            cfg, env, agent, replay, label, step_counter,
            strategy_model=strategy_model, groundtruth=groundtruth,
            switch_label=sw_label, switch_step=sw_step,
        )
        history.append({
            "episode": ep, "label": label, "reward": total_reward,
            "caught": int(caught), "steps": steps, "transitions": transitions,
        })
        episode_flags.append({"episode": ep, "switch_to": label, "n_transitions": transitions})
        if not silent and (ep % 10 == 0 or ep == int(rl["episodes"]) - 1):
            print(f"ep {ep:3d} {label:9s} reward={total_reward:6.1f} "
                  f"caught={int(caught)} steps={steps:4d} trans={transitions}")

    tag = f"{rl['replay']}-{'det' if not groundtruth else 'gt'}-seed{cfg['seed']}"
    os.makedirs("results", exist_ok=True)
    with open(f"results/train_{tag}.json", "w") as f:
        json.dump({"config": cfg, "history": history, "episode_flags": episode_flags}, f, indent=2)
    os.makedirs("data", exist_ok=True)
    agent.save(f"data/npc_{tag}.pt")
    if not silent:
        print(f"train done ({time.time()-t0:.1f}s) -> data/npc_{tag}.pt, results/train_{tag}.json")
    return {"tag": tag, "history": history, "agent": agent, "model_path": f"data/npc_{tag}.pt"}


def main():
    cfg = load_config()
    progress = train_with_config(cfg)


if __name__ == "__main__":
    main()