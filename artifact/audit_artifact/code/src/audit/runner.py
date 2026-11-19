"""Audit training/evaluation runner with per-episode provenance logs."""

import json
import os
import random
import time

import numpy as np
import torch

from ..env.chase_env import ChaseEnv, N_ACTIONS, _DIRS
from ..env.features import FeatureWindow
from ..env.strategy import LABEL_TO_ID
from ..rl.dqn_agent import DQNAgent
from ..rl.replay import ReplayMemory, StrategyPartitionedReplayMemory
from ..strategy.detector import FeatureHistory, StrategyClassifier
from ..strategy.transitions import CusumTransitionDetector
from .config import DATA_ROOT, LABELS, SWITCH_PAIRS, audit_config, checkpoint_path, run_dir
from .metrics import adaptation_summary, evaluate_summary


def seed_everything(seed, threads=1):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(threads))
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        pass


def load_detector(cfg, device="cpu"):
    path = cfg["detector"]["model_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"detector checkpoint not found: {path}; run the F7/detector training first"
        )
    model = StrategyClassifier(cfg).to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def make_replay(cfg):
    rl = cfg["rl"]
    legacy = bool(rl.get("legacy_global_rng", False))
    rng_seed = int(cfg["seed"]) + int(rl.get("replay_seed_offset", 0))
    if rl["replay"] == "single":
        return ReplayMemory(rl["buffer_capacity"], seed=rng_seed,
                            legacy_global_rng=legacy)
    return StrategyPartitionedReplayMemory(
        cfg, rl["buffer_capacity"], len(LABELS), seed=rng_seed,
        legacy_global_rng=legacy,
    )


def observed(cfg, env, base_obs, label_id):
    obs = env.observed_signal(
        base_obs, label_id, bool(cfg["rl"].get("cond_on_strategy", False))
    )
    if cfg["rl"].get("mask_oracle_onehot", False) and obs.shape[0] > 14:
        obs = obs.copy()
        obs[-len(LABELS):] = 0.0
    return obs


def switch_plan(episode, switch_steps):
    """Return (start_label, [(step, target_label), ...])."""
    start, partner = SWITCH_PAIRS[int(episode) % len(SWITCH_PAIRS)]
    plan = []
    targets = [partner, start, partner, start]
    for k, step in enumerate(switch_steps):
        plan.append((int(step), targets[k % len(targets)]))
    return start, plan


def seed_expert_demos_audit(cfg, env, replay, per_strategy=8, silent=False):
    cond = bool(cfg["rl"].get("cond_on_strategy", False))
    attack_range = float(cfg["combat"]["attack_range"])
    jitter = float(cfg["rl"].get("params_jitter", 0.0))
    added = 0
    for label in LABELS:
        for _ in range(int(per_strategy)):
            env.reset(player_strategy=label, params_jitter=jitter)
            label_id = LABEL_TO_ID[label]
            for _ in range(int(cfg["env"]["episode_len"])):
                base = env._obs()
                obs = observed(cfg, env, base, label_id)
                dvec = env.player.pos - env.npc_pos
                dn = float(np.linalg.norm(dvec))
                dirn = dvec / dn if dn > 1e-6 else np.zeros(2)
                best, best_dot = 0, float("-inf")
                for a in range(len(_DIRS)):
                    dot = float(np.dot(dirn, _DIRS[a]))
                    if a != 0 and dot > best_dot:
                        best_dot, best = dot, a
                action = best if dn > attack_range else best + len(_DIRS)
                next_base, reward, terminated, truncated, info = env.step(action)
                next_obs = observed(cfg, env, next_base, label_id)
                done = terminated or truncated
                replay.add((obs, action, float(reward), next_obs, 1.0 if done else 0.0), label_id)
                added += 1
                if done:
                    break
    if not silent:
        print(f"seeded {added} expert transitions")
    return added


def _detector_state(cfg, strategy_model):
    if strategy_model is None:
        return None, None, None
    fw = FeatureWindow(cfg["strategy"]["feature_window"])
    fh = FeatureHistory(cfg["detector"]["context_len"])
    cusum = CusumTransitionDetector(cfg, len(LABELS))
    return fw, fh, cusum


def _update_detector(cfg, env, prev_npc, prev_player, prev_info,
                     fw, fh, cusum, strategy_model):
    if strategy_model is None:
        return None, 0
    fw.update(FeatureWindow.extract_step(
        env.npc_pos, prev_npc, env.player.pos, prev_player,
        combat={
            "player_attack": prev_info.get("player_attacked", False),
            "player_hit": prev_info.get("player_damage_dealt", 0.0) > 0,
            "damage_dealt": prev_info.get("player_damage_dealt", 0.0),
            "player_hp_frac": prev_info.get("player_hp", 1.0)
            / (env.player.max_hp or 1.0),
        },
    ))
    feat = fw.features()
    if feat is None:
        return None, 0
    fh.add(feat)
    if not fh.ready():
        return None, 0
    det_label, probs = strategy_model.predict(fh.as_context())
    _, is_trans = cusum.update(probs)
    return det_label, int(is_trans)


def run_training_episode(cfg, env, agent, replay, episode, start_label, plan,
                         step_counter, strategy_model, f5_writer=None,
                         f5_trace=False):
    rl = cfg["rl"]
    env.reset(
        player_strategy=start_label,
        params_jitter=float(rl.get("params_jitter", 0.0)),
    )
    fw, fh, cusum = _detector_state(cfg, strategy_model)
    current_label = start_label
    pending = list(plan)

    prev_npc = env.npc_pos.copy()
    prev_player = env.player.pos.copy()
    prev_info = {}
    rewards = []
    caught = False
    transitions = 0
    det_labels = []
    q_means, q_maxes, losses = [], [], []
    update_rows = 0

    for step in range(int(cfg["env"]["episode_len"])):
        while pending and step >= pending[0][0]:
            _, new_label = pending.pop(0)
            env.set_player_strategy(new_label)
            current_label = new_label

        label_id = LABEL_TO_ID[current_label]
        det_label, n_trans = _update_detector(
            cfg, env, prev_npc, prev_player, prev_info, fw, fh, cusum,
            strategy_model,
        )
        if strategy_model is not None:
            transitions += n_trans
            if det_label is not None:
                label_id = det_label
        det_labels.append(label_id)

        base_obs = env._obs()
        obs = observed(cfg, env, base_obs, label_id)
        action = agent.act(obs, step_counter[0])

        next_base, reward, terminated, truncated, info = env.step(action)
        next_obs = observed(cfg, env, next_base, label_id)
        rewards.append(float(reward))
        caught = caught or bool(info["caught"])
        done = terminated or truncated

        replay.add((obs, action, float(reward), next_obs, 1.0 if done else 0.0), label_id)
        step_counter[0] += 1

        learn_every = max(1, int(rl.get("learn_every", 2)))
        if len(replay) >= int(rl["batch_size"]) and (step_counter[0] % learn_every == 0):
            batch = replay.sample(
                rl["batch_size"],
                current_strategy_id=label_id,
            )
            if batch:
                agent.update(batch)
                stats = dict(agent.last_stats)
                if stats:
                    q_means.append(stats["q_mean_abs"])
                    q_maxes.append(stats["q_max_abs"])
                    losses.append(stats["loss"])
                    if f5_trace and f5_writer is not None:
                        row = {
                            "episode": int(episode),
                            "step": int(step),
                            "label_id": int(label_id),
                            **stats,
                        }
                        f5_writer.write(json.dumps(row) + "\n")
                        update_rows += 1

        prev_npc = env.npc_pos.copy()
        prev_player = env.player.pos.copy()
        prev_info = info
        if done:
            break

    adapt = adaptation_summary(rewards, cfg["audit"]["switch_steps"],
                               roll_win=int(cfg["audit"].get("roll_win", 20)))
    return {
        "episode": int(episode),
        "start_label": start_label,
        "switches": [{"step": int(s), "to": t} for s, t in plan],
        "reward": float(np.sum(rewards)),
        "caught": int(caught),
        "steps": len(rewards),
        "transitions": int(transitions),
        "adaptation_status": adapt["status"],
        "adaptation_time": adapt["adaptation_time"],
        "adaptation_details": adapt["details"],
        "n_ok": adapt["n_ok"],
        "n_censored": adapt["n_censored"],
        "n_no_post": adapt["n_no_post"],
        "n_short_base": adapt["n_short_base"],
        "q_mean_abs": float(np.mean(q_means)) if q_means else None,
        "q_max_abs": float(np.max(q_maxes)) if q_maxes else None,
        "loss": float(np.mean(losses)) if losses else None,
        "n_updates": len(q_means),
        "f5_rows": update_rows,
    }


def train_audit(cfg, silent=False):
    seed_everything(cfg["seed"])
    t0 = time.time()
    out_dir = run_dir(cfg, "train")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path(cfg)), exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    env = ChaseEnv(cfg, seed=cfg["seed"])
    agent = DQNAgent(cfg, env.obs_dim(), N_ACTIONS)
    replay = make_replay(cfg)
    strategy_model = load_detector(cfg) if cfg["rl"]["strategy_source"] == "detector" else None

    n_demos = int(cfg["rl"].get("seed_demos", 0))
    if n_demos > 0:
        seed_expert_demos_audit(cfg, env, replay, per_strategy=n_demos, silent=silent)

    ep_path = os.path.join(out_dir, "episodes.jsonl")
    f5_path = os.path.join(out_dir, "f5_updates.jsonl")
    f5_handle = open(f5_path, "w") if cfg["rl"].get("f5_trace", False) else None
    history = []
    step_counter = [0]
    try:
        with open(ep_path, "w") as ep_handle:
            for ep in range(int(cfg["rl"]["episodes"])):
                start, plan = switch_plan(ep, cfg["audit"]["switch_steps"])
                row = run_training_episode(
                    cfg, env, agent, replay, ep, start, plan, step_counter,
                    strategy_model, f5_writer=f5_handle,
                    f5_trace=bool(cfg["rl"].get("f5_trace", False)),
                )
                history.append(row)
                ep_handle.write(json.dumps(row) + "\n")
                if not silent and (ep % 10 == 0 or ep == int(cfg["rl"]["episodes"]) - 1):
                    print(
                        f"ep {ep:3d} {start:9s} reward={row['reward']:7.1f} "
                        f"caught={row['caught']} steps={row['steps']:4d} "
                        f"q={row['q_mean_abs'] if row['q_mean_abs'] is None else round(row['q_mean_abs'], 2)}"
                    )
    finally:
        if f5_handle is not None:
            f5_handle.close()

    agent.save(checkpoint_path(cfg))
    summary = {
        "tag": cfg["audit"]["tag"],
        "config": cfg,
        "history": history,
        "checkpoint": checkpoint_path(cfg),
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if not silent:
        print(f"train done ({time.time() - t0:.1f}s) -> {checkpoint_path(cfg)}")
    return {"tag": cfg["audit"]["tag"], "history": history,
            "model_path": checkpoint_path(cfg), "summary": summary}


def load_agent(cfg, path):
    env = ChaseEnv(cfg, seed=cfg["seed"] + 1000)
    agent = DQNAgent(cfg, env.obs_dim(), N_ACTIONS)
    ck = torch.load(path, map_location=agent.device, weights_only=False)
    if "online" in ck:
        agent.online.load_state_dict(ck["online"])
        agent.target.load_state_dict(ck["target"])
    else:
        agent.online.load_state_dict(ck)
        agent.target.load_state_dict(agent.online.state_dict())
    return env, agent


def run_eval_episode(cfg, env, agent, replay, episode, start_label, plan,
                     step_counter, strategy_model, online=False):
    rl = cfg["rl"]
    env.reset(player_strategy=start_label)
    fw, fh, cusum = _detector_state(cfg, strategy_model)
    current_label = start_label
    pending = list(plan)

    prev_npc = env.npc_pos.copy()
    prev_player = env.player.pos.copy()
    prev_info = {}
    rewards = []
    caught = False
    transitions = 0

    for step in range(int(cfg["env"]["episode_len"])):
        while pending and step >= pending[0][0]:
            _, new_label = pending.pop(0)
            env.set_player_strategy(new_label)
            current_label = new_label

        label_id = LABEL_TO_ID[current_label]
        det_label, n_trans = _update_detector(
            cfg, env, prev_npc, prev_player, prev_info, fw, fh, cusum,
            strategy_model,
        )
        if strategy_model is not None:
            transitions += n_trans
            if det_label is not None:
                label_id = det_label

        base_obs = env._obs()
        obs = observed(cfg, env, base_obs, label_id)
        action = agent.greedy(obs)
        next_base, reward, terminated, truncated, info = env.step(action)
        next_obs = observed(cfg, env, next_base, label_id)
        rewards.append(float(reward))
        caught = caught or bool(info["caught"])
        done = terminated or truncated

        if online:
            replay.add((obs, action, float(reward), next_obs, 1.0 if done else 0.0),
                       label_id)
            if step % 4 == 3:
                batch = replay.sample(
                    rl["batch_size"], current_strategy_id=label_id
                )
                if len(batch) >= 8:
                    agent.update(batch)

        prev_npc = env.npc_pos.copy()
        prev_player = env.player.pos.copy()
        prev_info = info
        step_counter[0] += 1
        if done:
            break

    adapt = adaptation_summary(rewards, cfg["audit"]["switch_steps"],
                               roll_win=int(cfg["audit"].get("roll_win", 20)))
    return {
        "episode": int(episode),
        "start_label": start_label,
        "switches": [{"step": int(s), "to": t} for s, t in plan],
        "reward": float(np.sum(rewards)),
        "caught": int(caught),
        "steps": len(rewards),
        "transitions": int(transitions),
        "adaptation_status": adapt["status"],
        "adaptation_time": adapt["adaptation_time"],
        "adaptation_details": adapt["details"],
        "n_ok": adapt["n_ok"],
        "n_censored": adapt["n_censored"],
        "n_no_post": adapt["n_no_post"],
        "n_short_base": adapt["n_short_base"],
    }


def evaluate_audit(cfg, model_path, mode="offline", episodes=None,
                   seed_offset=1000, write=True, silent=False):
    if mode not in ("offline", "online"):
        raise ValueError("mode must be 'offline' or 'online'")
    seed_everything(int(cfg["seed"]) + int(seed_offset))
    episodes = int(episodes if episodes is not None else cfg["audit"]["eval_episodes"])
    env, agent = load_agent(cfg, model_path)
    replay = make_replay(cfg) if mode == "online" else None
    strategy_model = load_detector(cfg) if cfg["rl"]["strategy_source"] == "detector" else None

    out_dir = run_dir(cfg, "eval")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    step_counter = [0]
    ep_path = os.path.join(out_dir, f"{mode}.episodes.jsonl")
    handle = open(ep_path, "w") if write else None
    try:
        for ep in range(episodes):
            start, plan = switch_plan(ep, cfg["audit"]["switch_steps"])
            row = run_eval_episode(
                cfg, env, agent, replay, ep, start, plan, step_counter,
                strategy_model, online=(mode == "online"),
            )
            rows.append(row)
            if handle is not None:
                handle.write(json.dumps(row) + "\n")
    finally:
        if handle is not None:
            handle.close()

    summary = evaluate_summary(rows)
    summary.update({
        "tag": cfg["audit"]["tag"],
        "mode": mode,
        "model": model_path,
        "config": cfg,
        "results": rows,
    })
    if write:
        with open(os.path.join(out_dir, f"{mode}.summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    if not silent:
        print(
            f"[{cfg['audit']['tag']} {mode}] win={summary['win_rate']:.3f} "
            f"adapt={summary['mean_adaptation_time']} "
            f"included={summary['n_adapt_included']}/{summary['episodes']} "
            f"(no-post={summary['n_no_post']}, short-base={summary['n_short_base']}, "
            f"censored={summary['n_censored']})"
        )
    return summary
