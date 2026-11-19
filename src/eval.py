"""Evaluation: post-transition adaptation time and win/catch rates.

Usage:
    python -m src.eval --model data/npc_partitioned-gt-seed0.pt \
        [--eval.episodes 20] [--eval.online 1] [--eval.switch_step 120]

Runs episodes with a player strategy change at `switch_step` and measures:
  - adaptation_time : steps until post-switch rolling reward recovers to the
                      pre-switch mean (capped at episode length)
  - episode reward, catch (win) rate
  - number of detected transitions (detector mode only)
"""

import argparse
import json
import os

import numpy as np

from .config import load_config
from .env.chase_env import ChaseEnv, N_ACTIONS
from .env.features import FeatureWindow
from .env.strategy import LABEL_TO_ID
from .rl.dqn_agent import DQNAgent
from .rl.replay import ReplayMemory, StrategyPartitionedReplayMemory
from .strategy.detector import FeatureHistory, StrategyClassifier
from .strategy.transitions import CusumTransitionDetector

import torch

LABELS = ["aggressive", "defensive", "evasive", "balanced"]
ROLL_WIN = 20


def make_replay(cfg):
    rl = cfg["rl"]
    if rl["replay"] == "single":
        return ReplayMemory(rl["buffer_capacity"])
    return StrategyPartitionedReplayMemory(cfg, rl["buffer_capacity"], len(LABELS))


def load_model(cfg, agent, path):
    ck = torch.load(path, map_location=agent.device)
    if "online" in ck:
        agent.online.load_state_dict(ck["online"])
        agent.target.load_state_dict(ck["target"])
    else:
        agent.online.load_state_dict(ck)
    agent.target.load_state_dict(agent.online.state_dict())


def run_eval_episode(cfg, env, agent, replay, start_label, end_label, switch_step,
                     strategy_model, online=True, step_counter=(0,)):
    fw = FeatureWindow(cfg["strategy"]["feature_window"]) if strategy_model else None
    fh = FeatureHistory(cfg["detector"]["context_len"]) if strategy_model else None
    cusum = CusumTransitionDetector(cfg, len(LABELS)) if strategy_model else None

    env.reset(player_strategy=start_label)
    prev_npc = env.npc_pos.copy()
    prev_player = env.player.pos.copy()
    prev_info = {}
    rewards = []
    detected_trans = 0
    caught = False

    for step in range(cfg["env"]["episode_len"]):
        if step == switch_step:
            env.set_player_strategy(end_label)

        cond = bool(cfg["rl"].get("cond_on_strategy", False))

        label_id = LABEL_TO_ID[end_label if step >= switch_step else start_label]
        if strategy_model is not None:
            fw.update(FeatureWindow.extract_step(
                env.npc_pos, prev_npc, env.player.pos, prev_player,
                combat={
                    "player_attack": prev_info.get("player_attacked", False),
                    "player_hit": prev_info.get("player_damage_dealt", 0.0) > 0,
                    "damage_dealt": prev_info.get("player_damage_dealt", 0.0),
                    "player_hp_frac": prev_info.get("player_hp", 1.0) / (env.player.max_hp or 1.0),
                }))
            feat = fw.features()
            if feat is not None:
                fh.add(feat)
                if fh.ready():
                    det_label, probs = strategy_model.predict(fh.as_context())
                    _, is_trans = cusum.update(probs)
                    if is_trans:
                        detected_trans += 1
                    label_id = det_label

        base_obs = env._obs()
        obs = env.observed_signal(base_obs, label_id, cond)
        action = agent.greedy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = env.observed_signal(next_obs, label_id, cond)
        rewards.append(float(reward))
        caught = caught or bool(info["caught"])
        done = terminated or truncated

        if online:
            replay.add((obs, action, reward, next_obs, 1.0 if done else 0.0), label_id)
            updates = (step % 4 == 3)
            if updates:
                batch = replay.sample(cfg["rl"]["batch_size"], current_strategy_id=label_id)
                if len(batch) >= 8:
                    agent.update(batch)

        prev_npc = env.npc_pos.copy()
        prev_player = env.player.pos.copy()
        prev_info = info
        step_counter[0] += 1
        if done:
            break

    adaptation = measure_adaptation(rewards, switch_step)
    return {
        "reward": float(np.sum(rewards)),
        "caught": int(caught),
        "adaptation_time": adaptation,
        "transitions": detected_trans,
        "episode_return": float(np.sum(rewards)),
    }


def measure_adaptation(rewards, switch_step):
    """Steps after switch until rolling reward recovers to pre-switch mean."""
    rewards = np.asarray(rewards, dtype=np.float64)
    pre = rewards[max(0, switch_step - ROLL_WIN):switch_step]
    if len(pre) == 0:
        return len(rewards)
    target = float(pre.mean())
    for t in range(switch_step, len(rewards)):
        win = rewards[max(0, t - ROLL_WIN):t + 1]
        if win.mean() >= target:
            return t - switch_step
    return len(rewards) - switch_step


def evaluate_model(cfg, model_path, episodes=20, online=True, switch_step=100, seed_offset=1):
    env = ChaseEnv(cfg, seed=cfg["seed"] + seed_offset)
    agent = DQNAgent(cfg, env.obs_dim(), N_ACTIONS)
    load_model(cfg, agent, model_path)
    replay = make_replay(cfg)

    strategy_model = None
    if cfg["rl"].get("strategy_source") == "detector" and os.path.exists(cfg["detector"]["model_path"]):
        strategy_model = StrategyClassifier(cfg)
        state = torch.load(cfg["detector"]["model_path"], map_location="cpu")
        strategy_model.load_state_dict(state["state_dict"])
        strategy_model.eval()

    switch_pairs = [(LABELS[i], LABELS[(i + 1) % len(LABELS)]) for i in range(len(LABELS))]
    results = []
    step_counter = [0]
    total_caught = 0
    total_adapt = []
    for ep in range(episodes):
        start, end = switch_pairs[ep % len(switch_pairs)]
        r = run_eval_episode(cfg, env, agent, replay, start, end, switch_step,
                             strategy_model, online=online, step_counter=step_counter)
        r.update({"ep": ep, "from": start, "to": end})
        results.append(r)
        total_caught += r["caught"]
        total_adapt.append(r["adaptation_time"])

    summary = {
        "win_rate": total_caught / max(1, len(results)),
        "mean_adaptation_time": float(np.mean(total_adapt)),
        "median_adaptation_time": float(np.median(total_adapt)),
        "episodes": len(results),
        "results": results,
    }
    return summary


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--online", type=int, default=1)
    parser.add_argument("--switch_step", type=int, default=100)
    args, _ = parser.parse_known_args()

    summary = evaluate_model(cfg, args.model, episodes=args.episodes,
                             online=bool(args.online), switch_step=args.switch_step)
    tag = os.path.basename(args.model).replace(".pt", "")
    with open(f"results/eval_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"saved -> results/eval_{tag}.json")


if __name__ == "__main__":
    main()