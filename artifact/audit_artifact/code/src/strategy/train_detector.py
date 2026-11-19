"""Train (and validate) the strategy classifier on labeled telemetry.

Usage:
    python -m src.strategy.train_detector [--detector.epochs 80]

Generates labeled rollouts for each strategy (player stuck on one strategy per
episode), builds the feature context dataset, trains the MLP, and validates
against a held-out set, printing per-class accuracy.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import load_config
from ..env.chase_env import ChaseEnv, _DIRS
from ..env.features import FeatureWindow, feature_dim
from .detector import FeatureHistory, StrategyClassifier

LABELS = ["aggressive", "defensive", "evasive", "balanced"]


def flee_action(obs, n_actions=9):
    """Pick a discrete action that moves the NPC away from the player.

    Used during telemetry generation so the player keeps exhibiting its
    strategy (especially aggressive approach) without getting caught.
    """
    dx, dy = float(obs[0]), float(obs[1])  # player - npc direction
    target = np.asarray([-dx, -dy])        # flee direction for the NPC
    target = target / (np.linalg.norm(target) + 1e-9)
    best, best_dot = 0, -2.0
    for i, d in enumerate(_DIRS):
        if i == 0 or np.linalg.norm(d) < 1e-6:
            continue
        dot = float(np.dot(d, target))
        if dot > best_dot:
            best_dot, best = dot, i
    return best


def generate_dataset(cfg, seed=0, episodes_per_strategy=None, npc_behavior="gentle"):
    """Generate labeled telemetry for detector training.

    npc_behavior controls the reference NPC motion during data collection:
      - "gentle" : slow random walk (default). The player's strategy can
        actually manifest (aggressive closes, defensive keeps distance,
        evasive keeps a mid band), which matches gameplay realism better than
        a fully stationary or fully fleeing NPC.
      - "stationary" : NPC stays still (maximally clean signal).
      - "flee" : NPC runs away at full speed (strategies become indistinct).
    """
    d = cfg["detector"]
    episodes_per_strategy = episodes_per_strategy or d["rollout_episodes_per_strategy"]
    ep_len = d["rollout_episode_len"]
    window = cfg["strategy"]["feature_window"]
    ctx = d["context_len"]

    X, y = [], []
    rng = np.random.default_rng(seed)
    for li, label in enumerate(LABELS):
        for _ in range(episodes_per_strategy):
            env = ChaseEnv(cfg, seed=int(rng.integers(0, 2**31)))
            # keep both actors alive so combat telemetry flows for the window
            env.player.max_hp = env.player.hp = 1e9
            env.npc.max_hp = env.npc.hp = 1e9
            env.player_max_hp = env.npc_max_hp = 1e9
            env.reset(player_strategy=label)
            env.player.max_hp = env.player.hp = 1e9
            env.npc.max_hp = env.npc.hp = 1e9
            fw = FeatureWindow(window)
            fh = FeatureHistory(ctx)
            prev_player = env.player.pos.copy()
            prev_npc = env.npc_pos.copy()
            prev_info = {}
            for _ in range(ep_len):
                action = _generation_action(env, npc_behavior)
                _, _, _, truncated, info = env.step(action)
                fw.update(FeatureWindow.extract_step(
                    env.npc_pos, prev_npc, env.player.pos, prev_player,
                    combat={
                        "player_attack": info.get("player_attacked", False),
                        "player_hit": info.get("player_damage_dealt", 0.0) > 0,
                        "damage_dealt": info.get("player_damage_dealt", 0.0),
                        "player_hp_frac": info.get("player_hp", 10.0) / (env.player.max_hp or 1.0),
                    }))
                feat = fw.features()
                if feat is not None:
                    fh.add(feat)
                    if fh.ready():
                        X.append(np.concatenate(fh.as_context()))
                        y.append(li)
                prev_npc = env.npc_pos.copy()
                prev_player = env.player.pos.copy()
                if truncated:
                    break
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _generation_action(env, behavior):
    """Reference NPC action for telemetry generation."""
    if behavior == "stationary":
        return 0
    if behavior == "flee":
        return flee_action(env._obs())
    # gentle random walk (moves slower than the player so strategies show)
    rnd = env.rng
    choice = rnd.integers(0, 9)
    return int(choice)


def train_model(cfg, X, y, device="cpu"):
    d = cfg["detector"]
    rng = np.random.default_rng(cfg["seed"])
    perm = rng.permutation(len(y))
    n_val = max(int(0.2 * len(y)), 1)
    idx_tr, idx_va = perm[:-n_val], perm[-n_val:]

    Xtr, ytr = torch.from_numpy(X[idx_tr]), torch.from_numpy(y[idx_tr])
    Xva, yva = torch.from_numpy(X[idx_va]), torch.from_numpy(y[idx_va])

    model = StrategyClassifier(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(d["lr"]))
    loss_fn = nn.CrossEntropyLoss()

    bs = int(d["batch_size"])
    ds = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)

    for epoch in range(int(d["epochs"])):
        model.train()
        total, correct = 0.0, 0.0
        for xb, yb in ds:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += yb.size(0)
            correct += (logits.argmax(-1) == yb).sum().item()
        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == int(d["epochs"]):
            print(f"  epoch {epoch+1:3d}  train_acc={correct/total:.3f}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(Xva.to(device))
        pred = logits.argmax(-1).cpu().numpy()
    acc = float((pred == y[idx_va]).mean())
    per = {}
    for ci, name in enumerate(LABELS):
        mask = y[idx_va] == ci
        per[name] = float((pred[mask] == ci).mean()) if mask.sum() else float("nan")
    return model, acc, per


def save_model(model, cfg):
    path = cfg["detector"]["model_path"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, path)
    print(f"saved detector -> {path}")


def main():
    t0 = time.time()
    cfg = load_config()
    print("generating telemetry dataset ...")
    X, y = generate_dataset(cfg, seed=cfg["seed"])
    print(f"  dataset size: {X.shape[0]}  classes: {np.bincount(y).tolist()}")
    model, acc, per = train_model(cfg, X, y)
    print(f"validation accuracy: {acc:.3f}")
    for k, v in per.items():
        print(f"  {k:12s}: {v:.3f}")
    save_model(model, cfg)
    print(f"total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()